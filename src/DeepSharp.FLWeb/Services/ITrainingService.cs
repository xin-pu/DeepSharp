using DeepSharp.FLWeb.Models;

namespace DeepSharp.FLWeb.Services
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