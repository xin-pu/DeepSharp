namespace RLSharp.Core.Training
{
	public sealed record Transition<TState, TAction>(
		TState  State,
		TAction Action,
		float   Reward,
		TState  NextState,
		bool    IsTerminal);
}