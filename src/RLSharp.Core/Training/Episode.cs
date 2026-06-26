namespace RLSharp.Core.Training
{
	public sealed class Episode<TState, TAction>
	{
		private readonly List<Transition<TState, TAction>> _transitions = [];

		public IReadOnlyList<Transition<TState, TAction>> Transitions => _transitions;

		public float TotalReward => _transitions.Sum(t => t.Reward);

		public void Add(Transition<TState, TAction> transition)
		{
			_transitions.Add(transition);
		}
	}
}