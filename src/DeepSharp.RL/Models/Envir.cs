namespace DeepSharp.RL.Models
{
    /// <summary>
    ///     环境
    ///     提供观察 并给与奖励
    /// </summary>
    public abstract class Envir : ObservableObject
    {
        private string _name;
        private State? _state;
        private Reward? _reward;
        private List<State> _stateList;

        protected Envir()
            : this("Env")
        {
        }

        protected Envir(string name)
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
            set => SetProperty(ref _state, value);
            get => _state;
        }

        public Reward? Reward
        {
            set => SetProperty(ref _reward, value);
            get => _reward;
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
        public abstract State UpdateState(Action action);

        /// <summary>
        ///     根据环境状态计算奖励
        /// </summary>
        /// <param name="state"></param>
        /// <returns></returns>
        public abstract Reward GetReward(State state);
    }

    public class Action : ObservableObject
    {
        private DateTime _timeStamp;
        private torch.Tensor? _value;

        /// <summary>
        ///     奖励产生的时间戳
        /// </summary>
        public DateTime TimeStamp
        {
            set => SetProperty(ref _timeStamp, value);
            get => _timeStamp;
        }


        /// <summary>
        ///     奖励的张量格式
        /// </summary>
        public torch.Tensor? Value
        {
            set => SetProperty(ref _value, value);
            get => _value;
        }

        public override string ToString()
        {
            return $"{TimeStamp}\t{Value}";
        }
    }
}