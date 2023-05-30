using System.Reflection;

namespace DeepSharp.Dataset
{
    /// <summary>
    ///     Load DataSet by object-oriented inherit from torch.utils.data.Dataset
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class Dataset<T> : torch.utils.data.Dataset<T>
        where T : DataView
    {
        protected T[] AllData { set; get; }

        public Dataset(string path, char splitChar = '\t', bool hasHeader = true)
        {
            /// Step 1 Precheck
            File.Exists(path).Should().BeTrue($"File {path} should exist.");

            /// Step 2 Read Stream to DataTable or Array
            using var stream = new StreamReader(path);
            var allline = stream.ReadToEnd()
                .Split('\r', '\n')
                .Where(a => !string.IsNullOrEmpty(a))
                .ToList();
            if (hasHeader)
                allline.RemoveAt(0);
            var alldata = allline.Select(l => l.Split(splitChar).ToArray()).ToArray();

            var fieldDict = GetFieldDict(typeof(T));

            /// Step 3 According LoadColumnAttribute Change to Data
            AllData = alldata
                .Select(single => GetData(fieldDict, single))
                .ToArray();
            Count = AllData.Length;
        }

        public override long Count { get; }

        public override T GetTensor(long index)
        {
            return AllData[index];
        }


        #region protect function

        protected static Dictionary<PropertyInfo, StreamHeaderRange> GetFieldDict(Type type)
        {
            var fieldInfo = type.GetProperties()
                .Where(a => a.CustomAttributes
                    .Any(attributeData => attributeData.AttributeType == typeof(StreamHeaderAttribute)))
                .ToList();
            var dict = fieldInfo.ToDictionary(
                f => f,
                f => f.GetCustomAttribute<StreamHeaderAttribute>()!.StreamHeaderRange);
            return dict;
        }

        protected static T GetData(Dictionary<PropertyInfo, StreamHeaderRange> dict, string[] array)
        {
            var obj = (T) Activator.CreateInstance(typeof(T))!;
            dict.ToList().ForEach(p =>
            {
                var fieldInfo = p.Key;
                var range = p.Value;
                var type = fieldInfo.PropertyType;

                if (range.Min == range.Max)
                {
                    var field = Convert.ChangeType(array[range.Min], type);
                    fieldInfo.SetValue(obj, field);
                }
                else if (type.IsArray && range.Max >= range.Min)
                {
                    var len = range.Max - range.Min + 1;
                    var arr = Activator.CreateInstance(type, len);
                    Enumerable.Range(0, len).ToList().ForEach(i =>
                    {
                        var field = Convert.ChangeType(array[range.Min + i], type.GetElementType()!);
                        type.GetMethod("Set")?.Invoke(arr, new[] {i, field});
                    });
                    fieldInfo.SetValue(obj, arr);
                }
            });

            return obj;
        }

        #endregion
    }
}