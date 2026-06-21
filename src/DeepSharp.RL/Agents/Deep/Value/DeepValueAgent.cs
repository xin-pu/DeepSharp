using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents.Deep.Value;

/// <summary>
///     基于值网络的 Agent 基类
///     使用 Q 网络，策略为 argmax Q(s, a)
/// </summary>
public abstract class DeepValueAgent : DeepAgent
{
    protected DeepValueAgent(Environ<Space, Space> env, string name)
        : base(env, name)
    {
    }

    /// <summary>
    ///     Q 值网络
    /// </summary>
    public Module<torch.Tensor, torch.Tensor> Q { get; protected set; } = null!;

    /// <inheritdoc />
    public override Module<torch.Tensor, torch.Tensor> MainNet => Q;

    /// <summary>
    ///     策略动作：argmax Q(state, a)
    /// </summary>
    public override Act GetPolicyAct(torch.Tensor state)
    {
        var values       = Q.forward(state);
        var bestActIndex = torch.argmax(values).ToInt32();
        var actTensor    = torch.from_array(new[] { bestActIndex });
        return new Act(actTensor);
    }
}
