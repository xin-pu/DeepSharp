using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents;

/// <summary>
///     Agent 核心接口，定义所有智能体的基本契约
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
