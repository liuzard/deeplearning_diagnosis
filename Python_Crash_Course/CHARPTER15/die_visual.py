from Python_Crash_Course.CHARPTER15.die import Die
import pygal

die1=Die()
die2=Die()
die3=Die()


results=[]
for roll_num in range(1000):
    result=die1.roll()+die2.roll()+die3.roll()
    results.append(result)

frequencies=[]
for value in range(1,die1.num_sides+die2.num_sides+die3.num_sides+1):
    frequency=results.count(value)
    frequencies.append(frequency)
print(frequencies)

#对结果进行可视化
hist=pygal.Bar()
hist.title="result of rolling on three D6 1000 times"
hist.x_labels=['3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18']
hist.x_title="results"
hist.y_title="Frequency of result"

hist.add('D6+D6+D6',frequencies)
hist.render_to_file('die_visual.svg')