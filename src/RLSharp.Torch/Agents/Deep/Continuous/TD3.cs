using RLSharp.Torch.Environs;

namespace RLSharp.Torch.Agents.Deep.Continuous
{
	/// <summary>
	///     Twin Delayed DDPG algorithm boundary.
	/// </summary>
	public class TD3 : ContinuousControlAgent
	{
		public TD3(EnvironmentBase<Space, Space> env)
			: base(env, "TD3")
		{
			Critic2 = new Net(ObservationSize + ActionSize, 128, 1);
		}

		public Module<torch.Tensor, torch.Tensor> Critic2 { get; }
	}
}