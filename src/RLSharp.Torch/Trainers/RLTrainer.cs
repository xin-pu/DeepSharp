using RLSharp.Core;
using RLSharp.Torch.Agents;
using RLSharp.Torch.Environs;

namespace RLSharp.Torch.Trainers
{
	public class RLTrainer : Trainer
	{
		private TrainerCallback? callback;

		public RLTrainer(Agent agent)
		{
			Agent = agent;
		}

		public RLTrainer(Agent agent, Action<object> print)
		{
			Agent = agent;
			Print = print;
		}

		public Agent Agent { get; set; }

		public TrainerCallback? Callback
		{
			get => callback;
			set
			{
				callback = value;
				if (callback != null)
					callback.RLTrainer = this;
			}
		}

		public Action<object>? Print { get; set; }


		public override void Train()
		{
			Train(0f, 1000);
		}

		public virtual void Train(
			float             preReward,
			int               trainEpoch,
			string            saveFolder        = "",
			int               testEpisodes      = -1,
			int               testInterval      = 5,
			bool              autoSave          = false,
			CancellationToken cancellationToken = default)
		{
			OnTrainStart();

			try
			{
				var valEpoch = 0;
				foreach (var epoch in Enumerable.Range(1, trainEpoch))
				{
					cancellationToken.ThrowIfCancellationRequested();
					OnLearnStart(epoch);
					var outcome = Agent.Learn();
					OnLearnEnd(epoch, outcome);

					if (testEpisodes <= 0 || epoch % testInterval != 0)
						continue;

					valEpoch++;
					OnValStart(valEpoch);
					var episodes = Agent.RunEpisodes(testEpisodes);
					OnValStop(valEpoch, episodes);
					var valReward = episodes.Average(e => e.SumReward.Value);

					if (valReward < preReward)
						continue;

					if (autoSave)
					{
						OnSaveStart();
						Agent.Save(Path.Combine(saveFolder, $"[{Agent}]_{epoch}_{valReward:F2}.st"));
						OnSaveEnd();
					}

					break;
				}
			}
			finally
			{
				OnTrainEnd();
			}
		}


		public virtual void Train(RLTrainOption tp)
		{
			Train(tp.StopReward, tp.TrainEpoch, tp.SaveFolder, tp.ValEpisode, tp.ValInterval, tp.AutoSave);
		}

		public override void Val(int valEpoch)
		{
			var episodes  = Agent.RunEpisodes(valEpoch);
			var aveReward = episodes.Average(a => a.SumReward.Value);
			Print?.Invoke($"[Val] {valEpoch:D5}\tR:[{aveReward}]");
			foreach (var episode in episodes) Print?.Invoke(episode);
			OnValEnd(valEpoch, aveReward);
		}


		#region MyRegion

		protected override void OnTrainStart()
		{
			base.OnTrainStart();
			Print?.Invoke($"[{Agent}] start training.");
			Callback?.OnTrainStart();
		}

		protected override void OnTrainEnd()
		{
			base.OnTrainEnd();
			Print?.Invoke($"[{Agent}] stop training.");
			Callback?.OnTrainEnd();
		}


		protected virtual void OnLearnStart(int epoch)
		{
			Callback?.OnLearnStart(epoch);
		}


		protected virtual void OnLearnEnd(int epoch, LearnOutcome outcome)
		{
			Print?.Invoke($"[Tra]\t{epoch:D5}\t{outcome}");
			Callback?.OnLearnEnd(epoch, outcome);
		}

		protected override void OnValStart(int epoch)
		{
			base.OnValStart(epoch);
			Callback?.OnValStart(epoch);
		}

		protected virtual void OnValStop(int epoch, Episode[] episodes)
		{
			var aveReward = episodes.Average(a => a.SumReward.Value);
			Print?.Invoke($"[Val]\t{epoch:D5}\tE:{episodes.Length}:\tR:{aveReward:F4}");
			Callback?.OnValEnd(epoch, episodes);
		}

		protected override void OnSaveStart()
		{
			base.OnSaveStart();
			Callback?.OnSaveStart();
		}

		protected override void OnSaveEnd()
		{
			base.OnSaveEnd();
			Callback?.OnSaveEnd();
		}

		#endregion
	}
}