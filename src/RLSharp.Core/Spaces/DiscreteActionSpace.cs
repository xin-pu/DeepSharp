namespace RLSharp.Core.Spaces
{
	public sealed class DiscreteActionSpace<TAction> : IActionSpace<TAction>
	{
		public DiscreteActionSpace(IEnumerable<TAction> actions)
		{
			Actions = actions.ToArray();
			if (Actions.Count == 0)
				throw new ArgumentException("Action space must contain at least one action.", nameof(actions));
		}

		public IReadOnlyList<TAction> Actions { get; }

		public TAction Sample(Random random)
		{
			ArgumentNullException.ThrowIfNull(random);
			return Actions[random.Next(Actions.Count)];
		}
	}
}