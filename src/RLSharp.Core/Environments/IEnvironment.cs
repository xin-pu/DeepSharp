using RLSharp.Core.Spaces;

namespace RLSharp.Core.Environments
{
	public interface IEnvironment<TState, TAction>
	{
		string Name { get; }

		IActionSpace<TAction> ActionSpace { get; }

		TState Reset();

		StepResult<TState> Step(TAction action);
	}
}