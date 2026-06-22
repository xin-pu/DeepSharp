using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents.Deep.Policy;

/// <summary>
///     Policy-based neural network agent base class.
///     Uses PolicyNet to output action probability distribution, policy is multinomial sampling.
/// </summary>
public abstract class DeepPolicyAgent : DeepAgent
{
    protected DeepPolicyAgent(Environ<Space, Space> env, string name)
        : base(env, name)
    {
        PolicyNet = new PGN(ObservationSize, 128, ActionSize, DeviceType.CPU);
    }

    /// <summary>
    ///     Policy network.
    /// </summary>
    public Module<torch.Tensor, torch.Tensor> PolicyNet { get; protected set; }

    /// <inheritdoc />
    public override Module<torch.Tensor, torch.Tensor> MainNet => PolicyNet;

    /// <summary>
    ///     Policy action: multinomial sampling from softmax probabilities.
    /// </summary>
    public override Act GetPolicyAct(torch.Tensor state)
    {
        var probs    = PolicyNet.forward(state.unsqueeze(0)).squeeze(0);
        var actIndex = torch.multinomial(probs, 1, true).ToInt32();
        return new Act(torch.from_array(new[] { actIndex }));
    }
}