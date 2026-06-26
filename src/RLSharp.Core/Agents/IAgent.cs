using RLSharp.Core.Training;

namespace RLSharp.Core.Agents
{
	public interface IAgent<TState, TAction> : IPolicy<TState, TAction>
	{
		LearnResult Learn();

		void Save(string path);

		void Load(string path);
	}
}