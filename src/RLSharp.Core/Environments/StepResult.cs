namespace RLSharp.Core.Environments
{
	public sealed record StepResult<TState>(
		TState                                State,
		float                                 Reward,
		bool                                  IsTerminal,
		IReadOnlyDictionary<string, object?>? Info = null);
}