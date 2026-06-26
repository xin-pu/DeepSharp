using RLSharp.Torch.Environs;

namespace RLSharp.Torch.Agents.Deep.Continuous
{
	/// <summary>
	///     Base class for continuous-control actor-critic algorithms.
	/// </summary>
	public abstract class ContinuousControlAgent : DeepAgent
	{
		protected ContinuousControlAgent(EnvironmentBase<Space, Space> env, string name)
			: base(env, name)
		{
			Actor     = new Net(ObservationSize, 128, ActionSize);
			Critic    = new Net(ObservationSize + ActionSize, 128, 1);
			Optimizer = Adam(Actor.parameters(), 0.001f);
		}

		public Module<torch.Tensor, torch.Tensor> Actor { get; protected set; }

		public Module<torch.Tensor, torch.Tensor> Critic { get; protected set; }

		public override Module<torch.Tensor, torch.Tensor> MainNet => Actor;

		public override ActionValue GetPolicyAct(torch.Tensor state)
		{
			var action = Actor.forward(state.unsqueeze(0)).squeeze(0);
			return new ActionValue(action);
		}

		public override LearnOutcome Learn()
		{
			throw new NotSupportedException(
				$"{Name} requires a continuous-action environment adapter and replay buffer implementation.");
		}

		public override void Save(string path)
		{
			var dir  = Path.GetDirectoryName(path) ?? ".";
			var name = Path.GetFileNameWithoutExtension(path);
			Actor.save(Path.Combine(dir, $"{name}_actor.dat"));
			Critic.save(Path.Combine(dir, $"{name}_critic.dat"));
		}

		public override void Load(string path)
		{
			var dir  = Path.GetDirectoryName(path) ?? ".";
			var name = Path.GetFileNameWithoutExtension(path);
			Actor.load(Path.Combine(dir, $"{name}_actor.dat"));
			Critic.load(Path.Combine(dir, $"{name}_critic.dat"));
		}
	}
}