using RLSharp.Core.Agents;
using RLSharp.Core.Environments;
using RLSharp.Core.Spaces;
using RLSharp.Core.Training;
using RLSharp.Torch.Agents.Generic;
using RLSharp.Torch.Encoding;
using RLSharp.Torch.Examples.FrozenLake;

namespace RLSharp.Tests.RLTest.CoreApi
{
	public class GenericLibraryApiTest
	{
		[Fact]
		public void CoreTrainerCanTrainCustomEnvironmentWithoutTorchTypes()
		{
			var env     = new LineWorldEnvironment();
			var agent   = new QLearningAgent<int, MoveAction>(env, state => state.ToString(), 0.5f, seed: 42);
			var trainer = new Trainer<int, MoveAction>(agent);

			var result = trainer.Train(new TrainingOptions { MaxEpisodes = 100 });
			agent.Epsilon = 0;

			result.Episodes.Should().Be(100);
			result.Steps.Should().BeGreaterThan(0);
			agent.SelectAction(0).Should().Be(MoveAction.Right);
		}

		[Fact]
		public void DqnAgentCanTrainSaveLoadAndPredictWithoutEnvironmentAtInference()
		{
			var env     = new LineWorldEnvironment();
			var encoder = new DelegateStateEncoder<int>(3, EncodeLineWorld);
			var agent   = new DqnAgent<int, MoveAction>(env, encoder, epsilon: 0, maxStepsPerEpisode: 4, seed: 42);
			agent.Learn();

			var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.dat");
			try
			{
				agent.Save(path);
				var loaded = new DqnAgent<int, MoveAction>(env, encoder, epsilon: 0, maxStepsPerEpisode: 4, seed: 42);
				loaded.Load(path);

				IPolicy<int, MoveAction> policy = loaded.Policy;
				var                      action = policy.SelectAction(0);

				Enum.GetValues<MoveAction>().Should().Contain(action);
			}
			finally
			{
				if (File.Exists(path))
					File.Delete(path);
			}
		}

		[Fact]
		public void PpoAgentCanLearnFromTypedFrozenLakeEnvironment()
		{
			var env     = new FrozenLakeEnvironment(seed: 42, maxSteps: 8);
			var encoder = new DelegateStateEncoder<FrozenLakeState>(16, state => state.ToOneHot());
			var agent   = new PpoAgent<FrozenLakeState, FrozenLakeAction>(env, encoder, maxStepsPerEpisode: 8);

			var result = agent.Learn();

			result.Steps.Should().BeGreaterThan(0);
			result.Loss.Should().NotBeNull();
			float.IsFinite(result.Loss!.Value).Should().BeTrue();
		}

		private static float[] EncodeLineWorld(int state)
		{
			var encoded = new float[3];
			encoded[state] = 1f;
			return encoded;
		}

		private enum MoveAction
		{
			Left,
			Right
		}

		private sealed class LineWorldEnvironment : IEnvironment<int, MoveAction>
		{
			private int _state;

			public string Name => "LineWorld";

			public IActionSpace<MoveAction> ActionSpace { get; } =
				new DiscreteActionSpace<MoveAction>(Enum.GetValues<MoveAction>());

			public int Reset()
			{
				_state = 0;
				return _state;
			}

			public StepResult<int> Step(MoveAction action)
			{
				_state = action == MoveAction.Right
					? Math.Min(2, _state + 1)
					: Math.Max(0, _state - 1);

				var terminal = _state == 2;
				var reward   = terminal ? 1f : 0f;
				return new StepResult<int>(_state, reward, terminal);
			}
		}
	}
}