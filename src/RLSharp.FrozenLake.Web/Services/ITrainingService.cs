using RLSharp.FrozenLake.Web.Models;

namespace RLSharp.FrozenLake.Web.Services
{
	public interface ITrainingService
	{
		bool IsTraining { get; }

		Task StartTraining(TrainingConfig config, string connectionId);
		Task StopTraining(string          connectionId);
		Task RunDemo(string               connectionId);
		Task ResetEnvironment(string      connectionId);
	}
}