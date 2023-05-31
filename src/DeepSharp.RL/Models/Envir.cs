namespace DeepSharp.RL.Models
{
    /// <summary>
    ///     环境
    ///     提供观察 并给与奖励
    /// </summary>
    public abstract class Environ : ObservableObject
    {
        private string _name;
        private Observation _observation;
        private Reward _reward;
        private List<Observation> _observationList = new();


        protected Environ(string name)
        {
            _name = name;
            _observation = new Observation(torch.zeros(ObservationSpace));
        }

        public string Name
        {
            internal set => SetProperty(ref _name, value);
            get => _name;
        }

        public abstract int ActionSpace { set; get; }
        public abstract int ObservationSpace { set; get; }

        public Observation Observation
        {
            set => SetProperty(ref _observation, value);
            get => _observation;
        }

        public Reward Reward
        {
            set => SetProperty(ref _reward, value);
            get => _reward;
        }


        public List<Observation> ObservationList
        {
            internal set => SetProperty(ref _observationList, value);
            get => _observationList;
        }

        public int Life => ObservationList.Count;


        /// <summary>
        ///     恢复初始
        /// </summary>
        public void Reset()
        {
            ResetObservation();
            ObservationList.Clear();
        }

        public abstract void ResetObservation();

        /// <summary>
        ///     根据执行动作，更新环境 状态
        /// </summary>
        /// <param name="action"></param>
        /// <returns></returns>
        public abstract Observation UpdateState(Action action);

        /// <summary>
        ///     根据环境状态计算奖励
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public abstract Reward GetReward(Observation observation);
    }
}