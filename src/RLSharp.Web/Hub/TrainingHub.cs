using RLSharp.Web.Models;
using RLSharp.Web.Services;

namespace RLSharp.Web.Hub
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
			await _trainingService.StopTraining(Context.ConnectionId);
		}

		/// <summary>
		///     Client calls this to run a demo episode with current agent.
		/// </summary>
		public async Task RunDemo()
		{
			await _trainingService.RunDemo(Context.ConnectionId);
		}

		/// <summary>
		///     Client calls this to reset the environment without training.
		/// </summary>
		public async Task ResetEnv()
		{
			await _trainingService.ResetEnvironment(Context.ConnectionId);
		}

		public async Task ManualStep(string environmentType, string action)
		{
			await _trainingService.ManualStep(environmentType, action, Context.ConnectionId);
		}

		public override async Task OnDisconnectedAsync(Exception? exception)
		{
			await _trainingService.StopTraining(Context.ConnectionId);
			await base.OnDisconnectedAsync(exception);
		}
	}
}