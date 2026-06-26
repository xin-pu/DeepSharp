using RLSharp.Torch.Agents;
using RLSharp.Torch.Environs;

namespace RLSharp.Torch.Trainers
{
	public abstract class TrainerCallback
	{
		public RLTrainer RLTrainer { get; set; } = null!;

		public abstract void OnTrainStart();
		public abstract void OnTrainEnd();
		public abstract void OnLearnStart(int epoch);
		public abstract void OnLearnEnd(int   epoch, LearnOutcome outcome);
		public abstract void OnValStart(int   epoch);
		public abstract void OnValEnd(int     epoch, Episode[] episodes);
		public abstract void OnSaveStart();
		public abstract void OnSaveEnd();
	}
}