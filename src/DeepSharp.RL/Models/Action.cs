namespace DeepSharp.RL.Models
{
    public class Action : ObservableObject
    {
        private DateTime _timeStamp;
        private torch.Tensor _value = torch.zeros(1);

        public Action(torch.Tensor action)
        {
            Value = action;
            TimeStamp = DateTime.Now;
        }

        /// <summary>
        ///     奖励的张量格式
        /// </summary>
        public torch.Tensor Value
        {
            set => SetProperty(ref _value, value);
            get => _value;
        }

        /// <summary>
        ///     奖励产生的时间戳
        /// </summary>
        public DateTime TimeStamp
        {
            set => SetProperty(ref _timeStamp, value);
            get => _timeStamp;
        }


        public override string ToString()
        {
            return $"{TimeStamp}\t{Value}";
        }
    }
}