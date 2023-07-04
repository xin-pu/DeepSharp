using System.Text;

namespace DeepSharp.RL.Agents
{
    /// <summary>
    ///     State Value Function
    ///     V(s) = Cumulative reward
    /// </summary>
    public class VTable : IEquatable<VTable>
    {
        public VTable()
        {
            Return = new Dictionary<torch.Tensor, float>();
        }

        public Dictionary<torch.Tensor, float> Return { protected set; get; }

        protected List<torch.Tensor> StateKeys => Return.Keys.ToList();


        public float this[torch.Tensor state]
        {
            get => GetValue(state);
            set => SetValue(state, value);
        }

        private void SetValue(torch.Tensor state, float value)
        {
            Return[state] = value;
        }

        private float GetValue(torch.Tensor transit)
        {
            Return.TryAdd(transit, 0f);
            return Return[transit];
        }

        public static float operator -(VTable a, VTable b)
        {
            var keys = a.StateKeys;
            return keys.Select(k => Math.Abs(a[k] - b[k])).Max();
        }

        public bool Equals(VTable? other)
        {
            if (other == null) return false;
            if (other.StateKeys.Count != StateKeys.Count) return false;
            var res = StateKeys.All(key => !(Math.Abs(this[key] - other[key]) > 1E-2));
            return res;
        }


        public override string ToString()
        {
            var str = new StringBuilder();
            foreach (var keyValuePair in Return.Where(a => a.Value > 0))
                str.AppendLine($"{keyValuePair.Key.ToString(torch.numpy)}\t{keyValuePair.Value:F4}");
            return str.ToString();
        }
    }
}