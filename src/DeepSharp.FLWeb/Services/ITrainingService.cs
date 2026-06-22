using DeepSharp.FLWeb.Models;

namespace DeepSharp.FLWeb.Services;

public interface ITrainingService
{
    bool IsTraining { get; }

    Task StartTraining(TrainingConfig config, string connectionId);
    Task StopTraining();
    Task RunDemo();
    Task ResetEnvironment();
}
