using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents.Deep.Value
{
	/// <summary>
	///     Value-based neural network agent base class.
	///     Uses a Q network with policy: argmax Q(s, a).
	/// </summary>
	public abstract class DeepValueAgent : DeepAgent
	{
		protected DeepValueAgent(Environ<Space, Space> env, string name)
			: base(env, name)
		{
		}

		/// <summary>
		///     Q-value network.
		/// </summary>
		public Module<torch.Tensor, torch.Tensor> Q { get; protected set; } = null!;

		/// <inheritdoc />
		public override Module<torch.Tensor, torch.Tensor> MainNet => Q;

		/// <summary>
		///     Policy action: argmax Q(state, a).
		/// </summary>
		public override Act GetPolicyAct(torch.Tensor state)
		{
			var values       = Q.forward(state);
			var bestActIndex = torch.argmax(values).ToInt32();
			var actTensor    = torch.from_array(new[] { bestActIndex });
			return new Act(actTensor);
		}
	}
}