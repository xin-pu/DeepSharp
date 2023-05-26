namespace RL.Core
{
    /// <summary>
    ///     环境
    ///     提供观察 并给与奖励
    /// </summary>
    public abstract class Environment : ObservableObject
    {
        private string _name;
        private State? _state;
        private List<State> _stateList;

        protected Environment()
            : this("Env")
        {
        }

        protected Environment(string name)
        {
            _state = null;
            _name = name;
            _stateList = new List<State>();
        }

        public string Name
        {
            internal set => SetProperty(ref _name, value);
            get => _name;
        }

        public State? State
        {
            internal set => SetProperty(ref _state, value);
            get => _state;
        }


        public List<State> StateList
        {
            internal set => SetProperty(ref _stateList, value);
            get => _stateList;
        }

        public int Life => StateList.Count;

        /// <summary>
        ///     环境 根据当前当做更新状态
        ///     根据新状态，返回奖励
        /// </summary>
        /// <param name="Action"></param>
        /// <returns></returns>
        public Reward Step(Action action)
        {
            State = UpdateState(action);
            StateList.Add(State);
            var reward = GetReward(State);
            return reward;
        }

        /// <summary>
        ///     恢复初始
        /// </summary>
        public void Reset()
        {
            State = null;
            StateList.Clear();
        }

        /// <summary>
        ///     根据执行动作，更新环境 状态
        /// </summary>
        /// <param name="action"></param>
        /// <returns></returns>
        internal abstract State UpdateState(Action action);

        /// <summary>
        ///     根据环境状态计算奖励
        /// </summary>
        /// <param name="state"></param>
        /// <returns></returns>
        internal abstract Reward GetReward(State state);
    }
}