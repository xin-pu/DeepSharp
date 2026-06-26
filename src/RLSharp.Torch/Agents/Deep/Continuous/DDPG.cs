using RLSharp.Torch.Environs;

namespace RLSharp.Torch.Agents.Deep.Continuous
{
	/// <summary>
	///     Deep Deterministic Policy Gradient algorithm boundary.
	/// </summary>
	public class DDPG : ContinuousControlAgent
	{
		public DDPG(EnvironmentBase<Space, Space> env)
			: base(env, "DDPG")
		{
		}
	}
}
