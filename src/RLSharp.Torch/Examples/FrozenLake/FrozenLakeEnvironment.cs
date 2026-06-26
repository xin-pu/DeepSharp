using RLSharp.Core.Environments;
using RLSharp.Core.Spaces;

namespace RLSharp.Torch.Examples.FrozenLake
{
	public sealed class FrozenLakeEnvironment : IEnvironment<FrozenLakeState, FrozenLakeAction>
	{
		private readonly LakeCell[] _cells;
		private readonly Random     _random;
		private          int        _steps;

		public FrozenLakeEnvironment(int size = 4, int? seed = null, int maxSteps = 100)
		{
			if (size != 4)
				throw new ArgumentOutOfRangeException(nameof(size),
					"The sample FrozenLake map currently supports size 4.");

			Size     = size;
			MaxSteps = maxSteps;
			_random  = seed is null ? new Random() : new Random(seed.Value);
			_cells =
			[
				LakeCell.Start, LakeCell.Frozen, LakeCell.Frozen, LakeCell.Frozen,
				LakeCell.Frozen, LakeCell.Hole, LakeCell.Frozen, LakeCell.Hole,
				LakeCell.Frozen, LakeCell.Frozen, LakeCell.Frozen, LakeCell.Hole,
				LakeCell.Hole, LakeCell.Frozen, LakeCell.Frozen, LakeCell.Goal
			];
			ActionSpace  = new DiscreteActionSpace<FrozenLakeAction>(Enum.GetValues<FrozenLakeAction>());
			CurrentState = new FrozenLakeState(0, Size);
		}

		public int Size { get; }

		public int MaxSteps { get; }

		public FrozenLakeState CurrentState { get; private set; }

		public string Name => "FrozenLake";

		public IActionSpace<FrozenLakeAction> ActionSpace { get; }

		public FrozenLakeState Reset()
		{
			_steps       = 0;
			CurrentState = new FrozenLakeState(0, Size);
			return CurrentState;
		}

		public StepResult<FrozenLakeState> Step(FrozenLakeAction action)
		{
			_steps++;
			var nextIndex = Move(CurrentState.PlayerIndex, action);
			CurrentState = new FrozenLakeState(nextIndex, Size);

			var cell     = _cells[nextIndex];
			var terminal = cell is LakeCell.Hole or LakeCell.Goal || _steps >= MaxSteps;
			var reward   = cell                                             == LakeCell.Goal ? 1f : 0f;
			return new StepResult<FrozenLakeState>(CurrentState, reward, terminal);
		}

		private int Move(int index, FrozenLakeAction action)
		{
			var row = index / Size;
			var col = index % Size;
			(row, col) = action switch
			{
				FrozenLakeAction.Up    => (Math.Max(0, row      - 1), col),
				FrozenLakeAction.Down  => (Math.Min(Size        - 1, row + 1), col),
				FrozenLakeAction.Left  => (row, Math.Max(0, col - 1)),
				FrozenLakeAction.Right => (row, Math.Min(Size   - 1, col + 1)),
				_                      => (row, col)
			};

			return row * Size + col;
		}

		private enum LakeCell
		{
			Start,
			Frozen,
			Hole,
			Goal
		}
	}
}