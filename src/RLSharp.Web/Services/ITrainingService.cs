using RLSharp.Web.Models;

namespace RLSharp.Web.Services
{
	public interface ITrainingService
	{
		bool IsTraining { get; }

		Task StartTraining(TrainingConfig config, string connectionId);
		Task StopTraining(string          connectionId);
		Task RunDemo(string               connectionId);
		Task ResetEnvironment(string      connectionId);
		Task ManualStep(string            environmentType, string action, string connectionId);
	}
}