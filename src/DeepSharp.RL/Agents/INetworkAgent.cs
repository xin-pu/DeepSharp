namespace DeepSharp.RL.Agents;

/// <summary>
///     神经网络 Agent 接口，提供优化器访问和检查点功能
/// </summary>
public interface INetworkAgent
{
    torch.optim.Optimizer Optimizer { get; }

    void SaveCheckpoint(string dir);
    void LoadCheckpoint(string dir);
}
