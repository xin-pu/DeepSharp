using RLSharp.Core.Agents;
using RLSharp.Core.Environments;
using RLSharp.Core.Training;
using RLSharp.Torch.Encoding;
using RLSharp.Torch.Policies;

namespace RLSharp.Torch.Agents.Generic
{
	public sealed class DqnAgent<TState, TAction> : IAgent<TState, TAction>
		where TAction : notnull
	{
		private readonly DiscreteActionEncoder<TAction> _actionEncoder;
		private readonly IEnvironment<TState, TAction>  _environment;
		private readonly Random                         _random;

		public DqnAgent(
			IEnvironment<TState, TAction> environment,
			IStateEncoder<TState>         stateEncoder,
			int                           hiddenSize         = 128,
			float                         epsilon            = 0.2f,
			float                         gamma              = 0.99f,
			float                         learningRate       = 0.001f,
			int                           maxStepsPerEpisode = 1_000,
			int?                          seed               = null)
		{
			_environment       = environment  ?? throw new ArgumentNullException(nameof(environment));
			StateEncoder       = stateEncoder ?? throw new ArgumentNullException(nameof(stateEncoder));
			_actionEncoder     = new DiscreteActionEncoder<TAction>(environment.ActionSpace);
			_random            = seed is null ? new Random() : new Random(seed.Value);
			Epsilon            = epsilon;
			Gamma              = gamma;
			MaxStepsPerEpisode = maxStepsPerEpisode;

			Network   = new Net(StateEncoder.InputSize, hiddenSize, _actionEncoder.ActionCount);
			Optimizer = Adam(Network.parameters(), learningRate);
			Policy    = new TorchPolicy<TState, TAction>(Network, StateEncoder, _actionEncoder);
		}

		public float Epsilon { get; set; }

		public float Gamma { get; set; }

		public int MaxStepsPerEpisode { get; set; }

		public IStateEncoder<TState> StateEncoder { get; }

		public Module<torch.Tensor, torch.Tensor> Network { get; }

		public Optimizer Optimizer { get; }

		public TorchPolicy<TState, TAction> Policy { get; }

		public LearnResult Learn()
		{
			var           state       = _environment.Reset();
			var           totalReward = 0f;
			var           steps       = 0;
			torch.Tensor? lastLoss    = null;

			while (steps < MaxStepsPerEpisode)
			{
				var action = SelectExploratoryAction(state);
				var result = _environment.Step(action);

				using var stateTensor     = StateEncoder.Encode(state).unsqueeze(0);
				using var nextStateTensor = StateEncoder.Encode(result.State).unsqueeze(0);
				using var qValues         = Network.forward(stateTensor);
				using var nextQValues     = Network.forward(nextStateTensor).detach();
				using var predicted       = qValues[0, _actionEncoder.ToIndex(action)];
				using var target =
					torch.tensor(result.Reward + (result.IsTerminal ? 0 : Gamma * nextQValues.max().item<float>()));
				using var loss = functional.mse_loss(predicted, target);

				Optimizer.zero_grad();
				loss.backward();
				Optimizer.step();

				lastLoss?.Dispose();
				lastLoss = loss.detach();

				totalReward += result.Reward;
				steps++;
				state = result.State;
				if (result.IsTerminal)
					break;
			}

			var lossValue = lastLoss?.item<float>();
			lastLoss?.Dispose();
			return new LearnResult(steps, totalReward, lossValue);
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
			Network.load(path);
		}

		private TAction SelectExploratoryAction(TState state)
		{
			return _random.NextDouble() < Epsilon
				? _environment.ActionSpace.Sample(_random)
				: SelectAction(state);
		}
	}
}