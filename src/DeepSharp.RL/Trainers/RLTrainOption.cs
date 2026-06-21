namespace DeepSharp.RL.Trainers
{
	public struct RLTrainOption
	{
		public RLTrainOption()
		{
			TrainEpoch  = 0;
			StopReward  = 0;
			ValEpisode  = 0;
			ValInterval = 0;
			SaveFolder  = string.Empty;
			OutTimeSpan = TimeSpan.FromHours(1);
			AutoSave    = false;
		}

		public float StopReward { get; set; }

		public int TrainEpoch { get; set; }

		public int ValInterval { get; set; }

		public int ValEpisode { get; set; }

		public string SaveFolder { get; set; }

		public TimeSpan OutTimeSpan { get; set; }

		public bool AutoSave { get; set; }
	}
}