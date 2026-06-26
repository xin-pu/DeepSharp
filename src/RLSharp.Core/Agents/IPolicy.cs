namespace RLSharp.Core.Agents
{
	public interface IPolicy<TState, TAction>
	{
		TAction SelectAction(TState state);

		TAction Predict(TState state)
		{
			return SelectAction(state);
		}
	}
}