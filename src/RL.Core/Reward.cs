using CommunityToolkit.Mvvm.ComponentModel;
using TorchSharp;

namespace RL.Core
{
    /// <summary>
    ///     环境给出的奖励
    /// </summary>
    public class Reward : ObservableObject
    {
        private DateTime _timeStamp;
        private torch.Tensor? _value;

        public Reward(torch.Tensor value)
        {
            Value = value;
            TimeStamp = DateTime.Now;
        }

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