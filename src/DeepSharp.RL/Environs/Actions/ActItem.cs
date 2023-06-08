namespace DeepSharp.RL.Environs.Actions
{
    public class ActItem : ObservableObject
    {
        private int _actionID;
        private float _min;
        private float _max;
        private int _count;

        public ActItem(int actionID, float min, float max, int space)
        {
            ActionID = actionID;
            Max = max;
            Min = min;
            Count = space;
        }

        public int ActionID
        {
            internal set => SetProperty(ref _actionID, value);
            get => _actionID;
        }

        public float Max
        {
            internal set => SetProperty(ref _max, value);
            get => _max;
        }

        public float Min
        {
            internal set => SetProperty(ref _min, value);
            get => _min;
        }

        public int Count
        {
            internal set => SetProperty(ref _count, value);
            get => _count;
        }
    }
}