using DeepSharp.FLWeb.Models;
using DeepSharp.FLWeb.Services;

namespace DeepSharp.FLWeb.Hub
{
	/// <summary>
	///     SignalR hub for FrozenLake training communication.
	///     Client methods this hub pushes to:
	///     StepUpdate(GridState), EpisodeEnd(TrainingProgress),
	///     TrainingStarted(), TrainingStopped(), Error(string)
	/// </summary>
	public class TrainingHub : Microsoft.AspNetCore.SignalR.Hub
	{
		private readonly ITrainingService _trainingService;

		public TrainingHub(ITrainingService trainingService)
		{
			_trainingService = trainingService;
		}

		/// <summary>
		///     Client calls this to start training with given config.
		/// </summary>
		public async Task StartTraining(TrainingConfig config)
		{
			var connectionId = Context.ConnectionId;
			await _trainingService.StartTraining(config, connectionId);
		}

		/// <summary>
		///     Client calls this to stop training.
		/// </summary>
		public async Task StopTraining()
		{
			await _trainingService.StopTraining();
		}

		/// <summary>
		///     Client calls this to run a demo episode with current agent.
		/// </summary>
		public async Task RunDemo()
		{
			await _trainingService.RunDemo();
		}

		/// <summary>
		///     Client calls this to reset the environment without training.
		/// </summary>
		public async Task ResetEnv()
		{
			await _trainingService.ResetEnvironment();
		}
	}
}