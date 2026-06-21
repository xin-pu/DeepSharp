using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents.Deep.ActorCritic;

/// <summary>
///     Actor-Critic 双网络 Agent 基类
///     持有 PolicyNet (Actor) 和 ValueNet (Critic)
/// </summary>
public abstract class ActorCriticAgent : DeepAgent
{
    protected ActorCriticAgent(Environ<Space, Space> env, string name)
        : base(env, name)
    {
    }

    /// <summary>
    ///     策略网络（Actor）
    /// </summary>
    public Module<torch.Tensor, torch.Tensor> PolicyNet { get; protected set; } = null!;

    /// <summary>
    ///     值网络（Critic）
    /// </summary>
    public Module<torch.Tensor, torch.Tensor> ValueNet { get; protected set; } = null!;

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

    /// <summary>
    ///     保存双网络
    /// </summary>
    public override void Save(string path)
    {
        var dir = Path.GetDirectoryName(path) ?? ".";
        var name = Path.GetFileNameWithoutExtension(path);
        PolicyNet.save(Path.Combine(dir, $"{name}_policy.dat"));
        ValueNet.save(Path.Combine(dir, $"{name}_value.dat"));
    }

    /// <summary>
    ///     加载双网络
    /// </summary>
    public override void Load(string path)
    {
        var dir = Path.GetDirectoryName(path) ?? ".";
        var name = Path.GetFileNameWithoutExtension(path);
        PolicyNet.load(Path.Combine(dir, $"{name}_policy.dat"));
        ValueNet.load(Path.Combine(dir, $"{name}_value.dat"));
    }
}
