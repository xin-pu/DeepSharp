using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents;

/// <summary>
///     Core Agent interface defining the basic contract for all agents.
/// </summary>
public interface IAgent
{
    string Name { get; }
    torch.Device Device { get; }
    Environ<Space, Space> Environ { get; }

    LearnOutcome Learn();
    void Save(string path);
    void Load(string path);
    Act GetPolicyAct(torch.Tensor state);
}
