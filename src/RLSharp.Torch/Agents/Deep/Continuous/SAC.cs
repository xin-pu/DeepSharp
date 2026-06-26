using RLSharp.Torch.Environs;

namespace RLSharp.Torch.Agents.Deep.Continuous
{
	/// <summary>
	///     Soft Actor-Critic algorithm boundary.
	/// </summary>
	public class SAC : ContinuousControlAgent
	{
		public SAC(EnvironmentBase<Space, Space> env)
			: base(env, "SAC")
		{
			Critic2 = new Net(ObservationSize + ActionSize, 128, 1);
		}

		public Module<torch.Tensor, torch.Tensor> Critic2 { get; }
	}
}