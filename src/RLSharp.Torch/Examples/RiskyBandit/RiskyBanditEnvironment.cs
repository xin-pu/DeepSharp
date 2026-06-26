using RLSharp.Core.Environments;
using RLSharp.Core.Spaces;

namespace RLSharp.Torch.Examples.RiskyBandit
{
	public sealed class RiskyBanditEnvironment : IEnvironment<RiskyBanditState, RiskyBanditAction>
	{
		private readonly Random _random;
		private          int    _step;
		private          float  _totalReward;

		public RiskyBanditEnvironment(int? seed = null, int maxSteps = 100)
		{
			_random      = seed is null ? new Random() : new Random(seed.Value);
			MaxSteps     = maxSteps;
			ActionSpace  = new DiscreteActionSpace<RiskyBanditAction>(Enum.GetValues<RiskyBanditAction>());
			CurrentState = new RiskyBanditState(0, -1, 0);
		}

		public int MaxSteps { get; }

		public RiskyBanditState CurrentState { get; private set; }

		public string Name => "RiskyBandit";

		public IActionSpace<RiskyBanditAction> ActionSpace { get; }

		public RiskyBanditState Reset()
		{
			_step        = 0;
			_totalReward = 0;
			CurrentState = new RiskyBanditState(0, -1, 0);
			return CurrentState;
		}

		public StepResult<RiskyBanditState> Step(RiskyBanditAction action)
		{
			_step++;
			var reward = action switch
			{
				RiskyBanditAction.Safe    => _random.NextDouble() < 0.8 ? 1f : 0f,
				RiskyBanditAction.Neutral => _random.NextDouble() < 0.5 ? 2f : -1f,
				RiskyBanditAction.Risky   => _random.NextDouble() < 0.2 ? 8f : -5f,
				_                         => 0f
			};
			_totalReward += reward;
			CurrentState =  new RiskyBanditState(_step, (int)action, _totalReward);
			return new StepResult<RiskyBanditState>(CurrentState, reward, _step >= MaxSteps);
		}
	}
}