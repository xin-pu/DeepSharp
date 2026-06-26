using RLSharp.Core.Agents;
using RLSharp.Core.Environments;
using RLSharp.Core.Training;
using RLSharp.Torch.Encoding;
using RLSharp.Torch.Policies;

namespace RLSharp.Torch.Agents.Generic
{
	public sealed class PpoAgent<TState, TAction> : IAgent<TState, TAction>
		where TAction : notnull
	{
		private readonly DiscreteActionEncoder<TAction> _actionEncoder;
		private readonly IEnvironment<TState, TAction>  _environment;
		private readonly Random                         _random = new();

		public PpoAgent(
			IEnvironment<TState, TAction> environment,
			IStateEncoder<TState>         stateEncoder,
			int                           hiddenSize         = 128,
			float                         gamma              = 0.99f,
			float                         learningRate       = 0.001f,
			float                         clipEpsilon        = 0.2f,
			int                           maxStepsPerEpisode = 1_000)
		{
			_environment       = environment  ?? throw new ArgumentNullException(nameof(environment));
			StateEncoder       = stateEncoder ?? throw new ArgumentNullException(nameof(stateEncoder));
			_actionEncoder     = new DiscreteActionEncoder<TAction>(environment.ActionSpace);
			Gamma              = gamma;
			ClipEpsilon        = clipEpsilon;
			MaxStepsPerEpisode = maxStepsPerEpisode;

			PolicyNetwork = new PGN(StateEncoder.InputSize, hiddenSize, _actionEncoder.ActionCount);
			Optimizer     = Adam(PolicyNetwork.parameters(), learningRate);
			Policy        = new TorchPolicy<TState, TAction>(PolicyNetwork, StateEncoder, _actionEncoder);
		}

		public float Gamma { get; }

		public float ClipEpsilon { get; }

		public int MaxStepsPerEpisode { get; }

		public IStateEncoder<TState> StateEncoder { get; }

		public Module<torch.Tensor, torch.Tensor> PolicyNetwork { get; }

		public Optimizer Optimizer { get; }

		public TorchPolicy<TState, TAction> Policy { get; }

		public LearnResult Learn()
		{
			var states  = new List<TState>();
			var actions = new List<TAction>();
			var rewards = new List<float>();
			var state   = _environment.Reset();

			for (var step = 0; step < MaxStepsPerEpisode; step++)
			{
				var action = SampleAction(state);
				var result = _environment.Step(action);
				states.Add(state);
				actions.Add(action);
				rewards.Add(result.Reward);
				state = result.State;
				if (result.IsTerminal)
					break;
			}

			if (states.Count == 0)
				return new LearnResult(0, 0, 0);

			using var stateBatch = torch.stack(states.Select(s => StateEncoder.Encode(s)).ToArray());
			using var actionBatch = torch.tensor(actions.Select(_actionEncoder.ToIndex).Select(i => (long)i).ToArray(),
				torch.ScalarType.Int64).view(-1, 1);
			using var returnBatch = torch.tensor(DiscountedReturns(rewards), torch.ScalarType.Float32).view(-1, 1);
			using var oldLogProb  = torch.log(PolicyNetwork.forward(stateBatch)).gather(1, actionBatch).detach();

			Optimizer.zero_grad();
			using var logProb      = torch.log(PolicyNetwork.forward(stateBatch)).gather(1, actionBatch);
			using var ratio        = torch.exp(logProb     - oldLogProb);
			using var clippedRatio = torch.clamp(ratio, 1f - ClipEpsilon, 1f + ClipEpsilon);
			using var objective    = torch.min(ratio * returnBatch, clippedRatio * returnBatch);
			using var loss         = -objective.mean();
			loss.backward();
			Optimizer.step();

			return new LearnResult(states.Count, rewards.Sum(), loss.item<float>());
		}

		public TAction SelectAction(TState state)
		{
			return Policy.SelectAction(state);
		}

		public void Save(string path)
		{
			Policy.Save(path);
		}

		public void Load(string path)
		{
			PolicyNetwork.load(path);
		}

		private TAction SampleAction(TState state)
		{
			using var input = StateEncoder.Encode(state).unsqueeze(0);
			using var probs = PolicyNetwork.forward(input).squeeze(0);
			var       index = torch.multinomial(probs, 1, true).ToInt64();
			return _actionEncoder.FromIndex(index);
		}

		private float[] DiscountedReturns(IReadOnlyList<float> rewards)
		{
			var returns = new float[rewards.Count];
			var running = 0f;
			for (var i = rewards.Count - 1; i >= 0; i--)
			{
				running    = rewards[i] + Gamma * running;
				returns[i] = running;
			}

			return returns;
		}
	}
}