namespace DeepSharp.RL.Trainers
{
    public struct RLTrainOption
    {
        public RLTrainOption()
        {
            TrainEpoch = 0;
            StopReward = 0;
            ValEpisode = 0;
            ValInterval = 0;
            SaveFolder = string.Empty;
            OutTimeSpan = TimeSpan.FromHours(1);
            AutoSave = false;
        }

        public float StopReward { set; get; }
        public int TrainEpoch { set; get; }
        public int ValInterval { set; get; }
        public int ValEpisode { set; get; }
        public string SaveFolder { set; get; }
        public TimeSpan OutTimeSpan { set; get; }
        public bool AutoSave { set; get; }
    }
}