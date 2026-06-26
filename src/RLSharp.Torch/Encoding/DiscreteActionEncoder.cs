using RLSharp.Core.Spaces;

namespace RLSharp.Torch.Encoding
{
	public sealed class DiscreteActionEncoder<TAction>
		where TAction : notnull
	{
		private readonly Dictionary<TAction, int> _indices;

		public DiscreteActionEncoder(IActionSpace<TAction> actionSpace)
		{
			ArgumentNullException.ThrowIfNull(actionSpace);
			Actions = actionSpace.Actions.ToArray();
			if (Actions.Count == 0)
				throw new ArgumentException("Action space must contain at least one action.", nameof(actionSpace));

			_indices = Actions.Select((action, index) => (action, index))
				.ToDictionary(pair => pair.action, pair => pair.index);
		}

		public IReadOnlyList<TAction> Actions { get; }

		public long ActionCount => Actions.Count;

		public int ToIndex(TAction action)
		{
			if (!_indices.TryGetValue(action, out var index))
				throw new ArgumentException("Action does not belong to the configured action space.", nameof(action));

			return index;
		}

		public TAction FromIndex(long index)
		{
			if (index < 0 || index >= Actions.Count)
				throw new ArgumentOutOfRangeException(nameof(index));

			return Actions[(int)index];
		}
	}
}