<!DOCTYPE html>
<html>

<body class="stackedit">
  <div class="stackedit__html"><p>对研一时候做的一个项目进行简短的总结~<br>
背景：为某煤矿公司智能研究中心做一个智能检测：<a href="https://baike.baidu.com/item/%E7%9F%B8%E7%9F%B3/2192577?fr=aladdin">矸（gān）石</a>充填防碰撞的检测和预警。矸石充填就是捣实机不断把传动带送过来的细碎矸石给往后捣实。以往都是矿工手动操控捣实机，由于矿工经常误操作，会把捣实机抬得过高，撞断传送带。现在使用计算机视觉的方法，借助海康威视的ip摄像头，实现一个智能的防碰撞系统，降低人力成本和撞击的概率。<br>
<strong>总共有五个模块：视频流读入，语义分割及图像处理，设计输入输出，程序加密，程序封装</strong><br>
Demo：<a href="https://www.bilibili.com/video/BV1z44y1B7RH/">Video</a></p>
<p><img src="https://img-blog.csdnimg.cn/7d9d4644b8964db9985735df75f9da0a.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2htNzQ1NQ==,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述"><br>
<img src="https://img-blog.csdnimg.cn/b1933a0e77134887a75ddbda790b43d1.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2htNzQ1NQ==,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述"></p>
<h1><a id="_8"></a>语义分割识别模块</h1>
<p>在讨论设计方案的时候，提出两种方案：<br>
1.目标检测（Object Detection）：如YOLO,Faster R-CNN…<br>
2.语义分割（Semantic Segmentation)：如AlexNet,Bisenet…<br>
区别在于前者是物体的分类及定位，后者是像素级别的分类。根据实际需求，最终是需要计算捣实机构和传送带之间的视觉距离。目标检测的优点：计算距离的时候比较方便，只要预测是准确的，距离的计算即是上下两个bounding_box的坐标相减，但是目标检测一旦没有检测出来，一个检测物体就会”消失‘了，整个系统就失效了，即是一个非0即1的过程。语义分割的优点：像素级分割，会比目标检测的识别更加准确且细致。测距的实现，可以通过算法来进行弥补。所以最终选择的语义分割的方法。</p>
<h2><a id="1_14"></a>1.神经网络的选择</h2>
<p>基于工程有一定的实时性要求，所以网络需要尽可能的快速轻量。<br>
通过我的实测和论文作者提供的FPS、iou等数据，bisenet是相对满足本项目的一个网络。所以最终选择该语义分割网络。实验环境：NVIDIA 2080TI+CUDA10.1<br>
reference:<em>https://github.com/CoinCheung/BiSeNet</em><br>
paper: <em>https://arxiv.org/abs/1808.00897</em><br>
网络的运行和测试可以参考原作者的github</p>
<h2><a id="2_20"></a>2.读取视频流</h2>
<p>使用python-opencv拉流，读取实时视频通过ip地址+端口 以及用户名密码等。</p>
<pre><code class="prism language-python"><span class="token keyword">import</span> cv2
<span class="token comment"># user: admin</span>
<span class="token comment"># pwd: 12345</span>
<span class="token comment"># main: 主码流</span>
<span class="token comment"># ip: 192.168.1.64</span>
<span class="token comment"># Channels: 实时数据</span>
<span class="token comment"># 1： 通道</span>
cap <span class="token operator">=</span> cv2<span class="token punctuation">.</span>VideoCapture<span class="token punctuation">(</span><span class="token string">"rtsp://admin:12345@192.168.1.64/main/Channels/1"</span><span class="token punctuation">)</span>
<span class="token keyword">print</span> <span class="token punctuation">(</span>cap<span class="token punctuation">.</span>isOpened<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
<span class="token keyword">while</span> cap<span class="token punctuation">.</span>isOpened<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
	 success<span class="token punctuation">,</span>frame <span class="token operator">=</span> cap<span class="token punctuation">.</span>read<span class="token punctuation">(</span><span class="token punctuation">)</span>
	 cv2<span class="token punctuation">.</span>imshow<span class="token punctuation">(</span><span class="token string">"frame"</span><span class="token punctuation">,</span>frame<span class="token punctuation">)</span>
	 cv2<span class="token punctuation">.</span>waitKey<span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">)</span>
</code></pre>
<p>相当于视频的输入直接从ip摄像头中读取并直接输入神经网络。</p>
<h2><a id="3_40"></a>3.画轮廓+去噪</h2>
<p>轮廓的目的：①为了后续能够锁定捣实机和传送带，为后面测距做准备②不受到其他因素产生的误检所干扰。<br>
利用cv2.findContours()函数来查找检测物体的轮廓，需要注意的是cv2.findContours()函数接受的参数为二值图，即黑白的（不是灰度图），所以读取的图像要先转成灰度的，再转成二值图，具体操作为下面的代码（注释）。</p>
<pre><code class="prism language-python">out_vis_image <span class="token operator">=</span> helpers<span class="token punctuation">.</span>colour_code_segmentation<span class="token punctuation">(</span>output_image<span class="token punctuation">,</span> label_values<span class="token punctuation">)</span>
img <span class="token operator">=</span> cv2<span class="token punctuation">.</span>resize<span class="token punctuation">(</span>cv2<span class="token punctuation">.</span>cvtColor<span class="token punctuation">(</span>np<span class="token punctuation">.</span>uint8<span class="token punctuation">(</span>out_vis_image<span class="token punctuation">)</span><span class="token punctuation">,</span> cv2<span class="token punctuation">.</span>COLOR_RGB2BGR<span class="token punctuation">)</span><span class="token punctuation">,</span> <span class="token punctuation">(</span><span class="token number">1024</span><span class="token punctuation">,</span> <span class="token number">1024</span><span class="token punctuation">)</span><span class="token punctuation">,</span>interpolation<span class="token operator">=</span>cv2<span class="token punctuation">.</span>INTER_AREA<span class="token punctuation">)</span>
gray <span class="token operator">=</span> cv2<span class="token punctuation">.</span>cvtColor<span class="token punctuation">(</span>img<span class="token punctuation">,</span> cv2<span class="token punctuation">.</span>COLOR_BGR2GRAY<span class="token punctuation">)</span>  <span class="token comment"># 转换颜色空间</span>
ret<span class="token punctuation">,</span> thresh <span class="token operator">=</span> cv2<span class="token punctuation">.</span>threshold<span class="token punctuation">(</span>gray<span class="token punctuation">,</span> <span class="token number">5</span><span class="token punctuation">,</span> <span class="token number">100</span><span class="token punctuation">,</span> <span class="token number">0</span><span class="token punctuation">)</span> <span class="token comment">#检测轮廓</span>
image<span class="token punctuation">,</span> contours<span class="token punctuation">,</span> hier <span class="token operator">=</span> cv2<span class="token punctuation">.</span>findContours<span class="token punctuation">(</span>thresh<span class="token punctuation">,</span> cv2<span class="token punctuation">.</span>RETR_TREE<span class="token punctuation">,</span> cv2<span class="token punctuation">.</span>CHAIN_APPROX_NONE<span class="token punctuation">)</span>
cv2<span class="token punctuation">.</span>drawContours<span class="token punctuation">(</span>img<span class="token punctuation">,</span> contours<span class="token punctuation">,</span> i<span class="token punctuation">,</span> <span class="token punctuation">(</span><span class="token number">0</span><span class="token punctuation">,</span> <span class="token number">0</span><span class="token punctuation">,</span> <span class="token number">0</span><span class="token punctuation">)</span><span class="token punctuation">,</span> cv2<span class="token punctuation">.</span>FILLED<span class="token punctuation">)</span><span class="token comment">#绘制轮廓</span>
</code></pre>
<p>需要注意的是：opencv2返回两个值：contours：hierarchy。opencv3会返回三个值,分别是img, countours, hierarchy，我这里使用opencv3所以会返回三个参数。contours返回的是所有轮廓的像素点坐标(x,y)。<br>
这里还涉及到一个opencv返回轮廓的坑：一个python的np.ndarray（n维数组），<strong>假设轮廓有50个点，OpenCV返回的ndarray的维数是(50, 1, 2)，而不是我们认为的(100, 2)</strong>。所以需要特殊对数据处理一下才能利用轮廓点的坐标。在numpy的数组中，用逗号分隔的是轴的索引。举个例子，假设有如下的数组：</p>
<pre><code class="prism language-python">a <span class="token operator">=</span> np<span class="token punctuation">.</span>array<span class="token punctuation">(</span><span class="token punctuation">[</span><span class="token punctuation">[</span><span class="token punctuation">[</span><span class="token number">3</span><span class="token punctuation">,</span><span class="token number">4</span><span class="token punctuation">]</span><span class="token punctuation">]</span><span class="token punctuation">,</span> <span class="token punctuation">[</span><span class="token punctuation">[</span><span class="token number">1</span><span class="token punctuation">,</span><span class="token number">2</span><span class="token punctuation">]</span><span class="token punctuation">]</span><span class="token punctuation">,</span><span class="token punctuation">[</span><span class="token punctuation">[</span><span class="token number">5</span><span class="token punctuation">,</span><span class="token number">7</span><span class="token punctuation">]</span><span class="token punctuation">]</span><span class="token punctuation">,</span><span class="token punctuation">[</span><span class="token punctuation">[</span><span class="token number">3</span><span class="token punctuation">,</span><span class="token number">7</span><span class="token punctuation">]</span><span class="token punctuation">]</span><span class="token punctuation">,</span><span class="token punctuation">[</span><span class="token punctuation">[</span><span class="token number">1</span><span class="token punctuation">,</span><span class="token number">8</span><span class="token punctuation">]</span><span class="token punctuation">]</span><span class="token punctuation">]</span><span class="token punctuation">)</span> 
</code></pre>
<p>这里的shape是 (5, 1, 2)，想要取出轮廓坐标需要 a[:,0]。<br>
由于矿下环境复杂，捣实的过程环境很差，有煤渣覆盖和水汽等，一开始的预测效果有点差强人意，后期通过扩充数据集训练更拟合损失函数，精度有较大的提升。除此之外，借助opencv来对误检的噪点进行去除，只选中中间捣实机构的有效区域。<br>
去噪<br>
有了轮廓点的坐标就可以利用opencv的库函数contourArea求出轮廓的面积，，剔除小面积误检的轮廓，锁定中间区域的倒是机构和传送带，并使用高斯去噪操作，去噪之后的效果可以基本锁定倒是机构和传送带。</p>
<h2><a id="4_62"></a>4.测算距离</h2>
<p>利用轮廓处理后的坐标，由于区别于目标检测，语义分割没有规整的矩形框，所以需要进行坐标选点。<br>
捣实机构和传送带两者，传送带如果不受到撞击是相对静止的，而捣实机构是会动态变化的，所以确定捣实机构的点坐标更加的关键。<br>
依次在轮廓上选取四个点，并且取y坐标的最小值（相对人眼视角的最大值），将最小值(x1,y1)的向上延长，使用这个点的很坐标x得到在传动带内轮廓的对应坐标(x2,y2),然后x1-x2就可得到像素距离（<strong>需要注意的是：这里opencv的矩阵和我们所理解的横纵坐标是相反的，垂直方向是x，水平方向是y，左上角是（0，0）</strong>）。<br>
这块在后期测试的时候，遇到一个问题：在捣实机构向前推的时候，由于捣实机构上面会附带碎的矸石，所以会遮挡住传送带，这个时候是无法识别传送带的。<br>
解决方案：一旦捣实机构的坐标小于传送带的下沿坐标，把传动带的坐标给保存下来，也就是上一时刻的传送带坐标。</p>
<h1><a id="_69"></a>输入/输出</h1>
<p>因为需求的变化，需要同时对四个相机（总共有81个相机，81选4）下的捣实环境同时进行防碰撞检测。主程序在主目录的main.py，<br>
为了减小程序的耦合，更好的并行运行，将输入视频流和图像处理分割成了两个子函数，分别为readframe和ca()分别有四个逻辑相同的子函数。<br>
因为所有的处理（包括语义分割、图像处理、视频流读入、黑屏检测、程序加密）都需要在后台处理，所以前端需要留一个输入供用户选择控制的四个相机，以及检测系统需要随时输出每台相机下捣实机构与传送带的距离，输出给后面的PLC端控制。分别为in.txt，out.txt。所以主要涉及到怎么实时的修改输入输出文件。python有一个对文本的读写功能，可以满足需求。</p>
<pre><code class="prism language-python">f <span class="token operator">=</span> <span class="token builtin">open</span><span class="token punctuation">(</span><span class="token string">'in.txt'</span><span class="token punctuation">)</span>
urlgl <span class="token operator">=</span> f<span class="token punctuation">.</span>read<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">.</span>strip<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">.</span>split<span class="token punctuation">(</span><span class="token string">','</span><span class="token punctuation">)</span>
</code></pre>
<p>这里urlgl 设置的就是ip地址尾部的每个相机的区别。</p>
<pre><code class="prism language-python"><span class="token keyword">def</span> <span class="token function">savetxt_compact</span><span class="token punctuation">(</span>fname<span class="token punctuation">,</span> x<span class="token punctuation">,</span> fmt<span class="token operator">=</span><span class="token string">"%d"</span><span class="token punctuation">,</span> delimiter<span class="token operator">=</span><span class="token string">','</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token keyword">with</span> <span class="token builtin">open</span><span class="token punctuation">(</span>fname<span class="token punctuation">,</span> <span class="token string">'w+'</span><span class="token punctuation">)</span> <span class="token keyword">as</span> fh<span class="token punctuation">:</span>
        <span class="token keyword">for</span> row <span class="token keyword">in</span> x<span class="token punctuation">:</span>
            line <span class="token operator">=</span> delimiter<span class="token punctuation">.</span>join<span class="token punctuation">(</span><span class="token string">"0"</span> <span class="token keyword">if</span> value <span class="token operator">==</span> <span class="token number">0</span> <span class="token keyword">else</span> fmt <span class="token operator">%</span> value <span class="token keyword">for</span> value <span class="token keyword">in</span> row<span class="token punctuation">)</span>
            fh<span class="token punctuation">.</span>write<span class="token punctuation">(</span>line <span class="token operator">+</span> <span class="token string">'\n'</span><span class="token punctuation">)</span>
</code></pre>
<p>保存输出的像素距离。</p>
<h1><a id="_89"></a>程序加密</h1>
<p>使用的是tkinter设计的一个GUI，功能就是程序可以被指定运行指定时间，计时器到期后需要输入密码才能获得重新启动。相当于输入密码，又加了一段程序使用的时间周期。通过一个外部文件读取剩余时间，程序内部设计了一个类似于密码本的字典，外部显示的话是一个乱码，但是输入正确密码之后就会怎么指定的时间。一个数字对应一个字母，在外部显示的就是打乱的英文码。过程就类似于加密传输中的编码和解码。</p>
<pre><code class="prism language-python">dict1 <span class="token operator">=</span> <span class="token punctuation">{</span><span class="token string">'0'</span><span class="token punctuation">:</span> <span class="token string">'H'</span><span class="token punctuation">,</span> <span class="token string">'1'</span><span class="token punctuation">:</span> <span class="token string">'*'</span><span class="token punctuation">,</span> <span class="token string">'2'</span><span class="token punctuation">:</span> <span class="token string">'q'</span><span class="token punctuation">,</span> <span class="token string">'3'</span><span class="token punctuation">:</span> <span class="token string">'M'</span><span class="token punctuation">,</span> <span class="token string">'4'</span><span class="token punctuation">:</span> <span class="token string">'&amp;'</span><span class="token punctuation">,</span> <span class="token string">'5'</span><span class="token punctuation">:</span> <span class="token string">'d'</span><span class="token punctuation">,</span> <span class="token string">'6'</span><span class="token punctuation">:</span> <span class="token string">'W'</span><span class="token punctuation">,</span> <span class="token string">'7'</span><span class="token punctuation">:</span> <span class="token string">'K'</span><span class="token punctuation">,</span> <span class="token string">'8'</span><span class="token punctuation">:</span> <span class="token string">'x'</span><span class="token punctuation">,</span> <span class="token string">'9'</span><span class="token punctuation">:</span> <span class="token string">'#'</span><span class="token punctuation">}</span>
dict2 <span class="token operator">=</span> <span class="token punctuation">{</span><span class="token string">'0'</span><span class="token punctuation">:</span> <span class="token string">'n'</span><span class="token punctuation">,</span> <span class="token string">'1'</span><span class="token punctuation">:</span> <span class="token string">'B'</span><span class="token punctuation">,</span> <span class="token string">'2'</span><span class="token punctuation">:</span> <span class="token string">'c'</span><span class="token punctuation">,</span> <span class="token string">'3'</span><span class="token punctuation">:</span> <span class="token string">'k'</span><span class="token punctuation">,</span> <span class="token string">'4'</span><span class="token punctuation">:</span> <span class="token string">'F'</span><span class="token punctuation">,</span> <span class="token string">'5'</span><span class="token punctuation">:</span> <span class="token string">'j'</span><span class="token punctuation">,</span> <span class="token string">'6'</span><span class="token punctuation">:</span> <span class="token string">'^'</span><span class="token punctuation">,</span> <span class="token string">'7'</span><span class="token punctuation">:</span> <span class="token string">'L'</span><span class="token punctuation">,</span> <span class="token string">'8'</span><span class="token punctuation">:</span> <span class="token string">'s'</span><span class="token punctuation">,</span> <span class="token string">'9'</span><span class="token punctuation">:</span> <span class="token string">'%'</span><span class="token punctuation">}</span>
dict3 <span class="token operator">=</span> <span class="token punctuation">{</span><span class="token string">'0'</span><span class="token punctuation">:</span> <span class="token string">'R'</span><span class="token punctuation">,</span> <span class="token string">'1'</span><span class="token punctuation">:</span> <span class="token string">'r'</span><span class="token punctuation">,</span> <span class="token string">'2'</span><span class="token punctuation">:</span> <span class="token string">'S'</span><span class="token punctuation">,</span> <span class="token string">'3'</span><span class="token punctuation">:</span> <span class="token string">'$'</span><span class="token punctuation">,</span> <span class="token string">'4'</span><span class="token punctuation">:</span> <span class="token string">'Y'</span><span class="token punctuation">,</span> <span class="token string">'5'</span><span class="token punctuation">:</span> <span class="token string">'Z'</span><span class="token punctuation">,</span> <span class="token string">'6'</span><span class="token punctuation">:</span> <span class="token string">'A'</span><span class="token punctuation">,</span> <span class="token string">'7'</span><span class="token punctuation">:</span> <span class="token string">'a'</span><span class="token punctuation">,</span> <span class="token string">'8'</span><span class="token punctuation">:</span> <span class="token string">'U'</span><span class="token punctuation">,</span> <span class="token string">'9'</span><span class="token punctuation">:</span> <span class="token string">'i'</span><span class="token punctuation">}</span>
</code></pre>
<h1><a id="_97"></a>程序封装</h1>
<p>因为企业的电脑是windows平台，且没有python和深度学习的环境，并且要求需要有可移植性，所以程序要整个封装成一个可执行文件。<br>
使用的是外置的Pyinstaller模块，可以直接pip安装。</p>
<pre><code class="prism language-python">pip install pyinstaller
</code></pre>
<p>基本语法：</p>
<pre><code class="prism language-python">PyInstaller <span class="token operator">-</span>F <span class="token operator">-</span>w <span class="token operator">-</span>i  xxx<span class="token punctuation">.</span>ico dev<span class="token punctuation">.</span>py <span class="token operator">-</span><span class="token operator">-</span>hidden<span class="token operator">-</span><span class="token keyword">import</span><span class="token operator">=</span>pandas<span class="token punctuation">.</span>_libs<span class="token punctuation">.</span>tslibs<span class="token punctuation">.</span>timedeltas
</code></pre>
<p>-F 指只生成一个exe文件，不生成其他dll文件<br>
-w 不弹出交互窗口,如果你想程序运行的时候，与程序进行交互，则不加该参数<br>
-i 设定程序图标 ，其后面的xxx.ico文件就是程序小图标<br>
dev.py 要打包的程序，如果你不是在dev.py同一级目录下执行的打包命令，这里得写上dev.py的路径地址<br>
–hidden-import=pandas._libs.tslibs.timedeltas 隐藏相关模块的引用</p>
<p>你在哪个目录下执行的命令，默认打包完成的文件或者文件夹就在该目录，需要特别注意的是如果程序是在虚拟环境下开发的，封装也需要在虚拟环境下执行，毕竟封装包括一些依赖库的封装。<br>
py文件运行没问题，不代表你打包后的文件运行就没问题，在windows下测试时，可以使用windows下的powershell来运行，这样就不会闪退了。封装真的是一个很痛苦，很折腾的事情！！CUDA版本也需要匹配，包括一些细小的版本区别。</p>
<h2><a id="_116"></a>后记</h2>
<p>本项目目前已经申请专利《一种基于深度学习的矸石充填捣实机构防碰撞应用方法》<br>
请勿转载~<br>
这个项目介绍已经同步到CSDN https://blog.csdn.net/hm7455/article/details/119064158</p>
</div>
</body>

</html>
