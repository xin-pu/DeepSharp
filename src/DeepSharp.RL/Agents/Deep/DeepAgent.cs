using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents.Deep;

/// <summary>
///     神经网络 Agent 基类
///     提供优化器、损失函数、主网络和默认的 Save/Load 实现
/// </summary>
public abstract class DeepAgent : Agent, INetworkAgent
{
    protected DeepAgent(Environ<Space, Space> env, string name)
        : base(env, name)
    {
    }

    /// <summary>
    ///     优化器
    /// </summary>
    public Optimizer Optimizer { get; protected set; } = null!;

    /// <summary>
    ///     损失函数
    /// </summary>
    public Loss<torch.Tensor, torch.Tensor, torch.Tensor> Loss { get; protected set; } = null!;

    /// <summary>
    ///     主网络（子类重写以暴露对应的网络模块）
    /// </summary>
    public abstract Module<torch.Tensor, torch.Tensor> MainNet { get; }

    /// <summary>
    ///     默认保存：保存主网络
    /// </summary>
    public override void Save(string path)
    {
        if (File.Exists(path)) File.Delete(path);
        MainNet.save(path);
    }

    /// <summary>
    ///     默认加载：加载主网络
    /// </summary>
    public override void Load(string path)
    {
        MainNet.load(path);
    }

    /// <summary>
    ///     保存检查点（模型 + 优化器状态）
    /// </summary>
    public virtual void SaveCheckpoint(string dir)
    {
        Directory.CreateDirectory(dir);
        var modelPath = Path.Combine(dir, "model.dat");
        Save(modelPath);
    }

    /// <summary>
    ///     加载检查点（模型 + 优化器状态）
    /// </summary>
    public virtual void LoadCheckpoint(string dir)
    {
        var modelPath = Path.Combine(dir, "model.dat");
        Load(modelPath);
    }
}
