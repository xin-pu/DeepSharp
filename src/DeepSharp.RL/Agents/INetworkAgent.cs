namespace DeepSharp.RL.Agents;

/// <summary>
///     Neural network Agent interface, providing optimizer access and checkpoint functionality.
/// </summary>
public interface INetworkAgent
{
    torch.optim.Optimizer Optimizer { get; }

    void SaveCheckpoint(string dir);
    void LoadCheckpoint(string dir);
}
