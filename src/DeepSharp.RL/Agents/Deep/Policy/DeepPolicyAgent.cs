using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents.Deep.Policy;

/// <summary>
///     基于策略网络的 Agent 基类
///     使用 PolicyNet 输出动作概率分布，策略为 multinomial 采样
/// </summary>
public abstract class DeepPolicyAgent : DeepAgent
{
    protected DeepPolicyAgent(Environ<Space, Space> env, string name)
        : base(env, name)
    {
        PolicyNet = new PGN(ObservationSize, 128, ActionSize, DeviceType.CPU);
    }

    /// <summary>
    ///     策略网络
    /// </summary>
    public Module<torch.Tensor, torch.Tensor> PolicyNet { get; protected set; }

    /// <inheritdoc />
    public override Module<torch.Tensor, torch.Tensor> MainNet => PolicyNet;

    /// <summary>
    ///     策略动作：按 softmax 概率采样
    /// </summary>
    public override Act GetPolicyAct(torch.Tensor state)
    {
        var probs    = PolicyNet.forward(state.unsqueeze(0)).squeeze(0);
        var actIndex = torch.multinomial(probs, 1, true).ToInt32();
        return new Act(torch.from_array(new[] { actIndex }));
    }
}