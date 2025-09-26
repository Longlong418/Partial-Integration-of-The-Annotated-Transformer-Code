from Model_Architecture import *
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src.to(device)
        self.src_mask = (src != pad).unsqueeze(-2).to(device)
        if tgt is not None:
            self.tgt = tgt[:, :-1].to(device)
            self.tgt_y = tgt[:, 1:].to(device)
            self.tgt_mask = self.make_std_mask(self.tgt, pad).to(device)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask
    

class TrainState:
    """Track number of steps, examples, and tokens processed"""
    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed

def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss
    

def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )



class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())
    


def data_gen(V, batch_size, nbatches,device):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))#随机生成整数，范围 [1, V-1]
        data[:, 0] = 1 #每个seq的起始位置设置为1，表示<BOS>
        src = data.requires_grad_(False).clone().detach().to(device)
        tgt = data.requires_grad_(False).clone().detach().to(device)

        yield Batch(src, tgt, 0)


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)##返回最大值和最大值所在的索引，我们只需要索引
        next_word = next_word[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src).fill_(next_word)], dim=1
        )
    return ys

class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None

def example_simple_model(save_path="simple_model.pt", num_epochs=30, batch_size=80):
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
        ),
    )

    train_losses = []

    for epoch in range(num_epochs):
        # ------------------
        # 训练
        # ------------------
        model.train()
        train_loss, _ = run_epoch(
            data_gen(V, batch_size, 20, device=device),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )
        train_losses.append(train_loss.item())# tensor->numpy

        # ------------------
        # 验证
        # ------------------
        model.eval()
        val_loss=run_epoch(
            data_gen(V, batch_size, 5, device=device),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )[0]

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss}")
    # ------------------
    # 保存模型
    # ------------------
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # ------------------
    # 可视化训练 loss
    # ------------------
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs+1), train_losses, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()



def load_and_decode(model_class, save_path, src_seq, device='cpu', max_len=None, start_symbol=0):
    """
    加载已保存的模型，并对输入序列进行贪婪解码
    参数:
        model_class: make_model 函数或模型类（需要能实例化模型）
        save_path: 已保存模型的路径
        src_seq: 输入序列 (list 或 tensor)
        device: 'cpu' 或 'cuda'
        max_len: 解码最大长度，如果 None 则取 src_seq 长度
        start_symbol: 解码起始符号

    返回:
        decoded tensor
    """
    # 假设模型的词表大小与训练时相同，这里示例 V=11, N=2 层
    V = 11
    model = model_class(V, V, N=2).to(device)
    
    # 加载保存的模型参数
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    
    # 构建输入 tensor
    if not isinstance(src_seq, torch.Tensor):
        src = torch.LongTensor([src_seq]).to(device)
    else:
        src = src_seq.to(device)
    if max_len is None:
        max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len).to(device)
    # 贪婪解码
    decoded = greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=start_symbol)
    
    return decoded

if __name__ == "__main__":
    # example_simple_model()
    decoded = load_and_decode(
    model_class=make_model,
    save_path="simple_model.pt",
    src_seq=[0, 2, 3, 7, 6,6,9, 7, 8, 9],
    device=device,
    start_symbol=0
)
print("Greedy decode result:", decoded)



