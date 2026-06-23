using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents.Deep.ActorCritic
{
	/// <summary>
	///     Actor-Critic dual-network agent base class.
	///     Holds PolicyNet (Actor) and ValueNet (Critic).
	/// </summary>
	public abstract class ActorCriticAgent : DeepAgent
	{
		protected ActorCriticAgent(Environ<Space, Space> env, string name)
			: base(env, name)
		{
		}

		/// <summary>
		///     Policy network (Actor).
		/// </summary>
		public Module<torch.Tensor, torch.Tensor> PolicyNet { get; protected set; } = null!;

		/// <summary>
		///     Value network (Critic).
		/// </summary>
		public Module<torch.Tensor, torch.Tensor> ValueNet { get; protected set; } = null!;

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

		/// <summary>
		///     Save both networks.
		/// </summary>
		public override void Save(string path)
		{
			var dir  = Path.GetDirectoryName(path) ?? ".";
			var name = Path.GetFileNameWithoutExtension(path);
			PolicyNet.save(Path.Combine(dir, $"{name}_policy.dat"));
			ValueNet.save(Path.Combine(dir, $"{name}_value.dat"));
		}

		/// <summary>
		///     Load both networks.
		/// </summary>
		public override void Load(string path)
		{
			var dir  = Path.GetDirectoryName(path) ?? ".";
			var name = Path.GetFileNameWithoutExtension(path);
			PolicyNet.load(Path.Combine(dir, $"{name}_policy.dat"));
			ValueNet.load(Path.Combine(dir, $"{name}_value.dat"));
		}
	}
}