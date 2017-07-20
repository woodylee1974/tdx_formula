TDX Formula (Python) Version 1.0.00

Copyright (C) 2017 Woody.Lee All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
The name of the Woody Lee may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

---

# Introduction
This TDX Formula implements a domain-language for stock analysis, which is used in most of Chinese stock
analysis client application.
For example, a MACD indicator looks like:
```
text = ''''
       DIF:EMA(CLOSE,SHORT)-EMA(CLOSE,LONG);
       DEA:EMA(DIF,MID);
       MACD:(DIF-DEA)*2,COLORSTICK;
''''
```

You may use this formula like:
```
formula = Formula(text)
result = formula.evaluate(pa)
macd = result['MACD']
...
```
Then you may use 'macd' do what you want.

# Features
This TDX Formula implements the following functions:
| HHV         | 取N日内的最高价                                                  |   |   |   |
|-------------|------------------------------------------------------------------|---|---|---|
| LLV         | 取N日内的最低价                                                  |   |   |   |
| SUM         | 取N日内的和                                                      |   |   |   |
| REF         | 取N日前的价格                                                    |   |   |   |
| CROSS       |  (self.CROSS, r'在%s上穿%s时触发', 2),                           |   |   |   |
| NOT         |  (self.NOT, r'对%s取反逻辑', 1),                                 |   |   |   |
| IF          |  (self.IF, r'IF(%s, %s, %s)', 3),                                |   |   |   |
| IFF         |  (self.IF, r'IFF(%s, %s, %s)', 3),                               |   |   |   |
| EVERY       |  (self.EVERY, r'%s周期内均满足%s', 2),                           |   |   |   |
| EXIST       |  (self.EXIST, r'EXIST(%s, %s)', 2),                              |   |   |   |
| STD         |  (self.STD, r'%s周期%s的样本标准差', 2),                         |   |   |   |
| VAR         |  (self.VAR, r'%s周期%s的样本方差', 2),                           |   |   |   |
| STDP        |  (self.STDP, r'%s周期%s的总体标准差', 2),                        |   |   |   |
| VARP        |  (self.VARP, r'%s周期%s的总体方差', 2),                          |   |   |   |
| MAX         |  (self.MAX, r'取最大值(%s, %s)', 2),                             |   |   |   |
| MIN         |  (self.MIN, r'取最小值(%s, %s)', 2),                             |   |   |   |
| COUNT       |  (self.COUNT, r'满足条件%s在统计周期%s内的个数', 2),             |   |   |   |
| ABS         |  (self.ABS, r'%s的绝对值', 1),                                   |   |   |   |
| SQRT        |  (self.SQRT, r'%s的平方根', 1),                                  |   |   |   |
| POW         |  (self.POW, r'%s的%s次方', 2),                                   |   |   |   |
| LOG         |  (self.LOG, r'%s的对数', 1),                                     |   |   |   |
| CONST       |  (self.CONST, r'%s的最后值为常量', 1),                           |   |   |   |
| INSIST      |  (self.INSIST, r'%s在周期%s到周期%s全为真', 3),                  |   |   |   |
| LAST        |  (self.INSIST, r'%s在周期%s到周期%s全为真', 3),                  |   |   |   |
| FILTER      |  (self.FILTER, r'过滤%s连续出现的%s个信号', 2),                  |   |   |   |
| BARSLAST    |  (self.BARSLAST, r'满足条件%s到当前的周期数', 1),                |   |   |   |
| AVEDEV      |  (self.AVEDEV, r'%s的周期%s的平均绝对偏差', 2),                  |   |   |   |
| MA          |  (self.MA, r'%s的%s日简单移动平均', 2),                          |   |   |   |
| EMA         |  (self.EMA, r'%s的%s日指数移动平均', 2),                         |   |   |   |
| EXPEMA      |  (self.EMA, r'%s的%s日指数移动平均', 2),                         |   |   |   |
| MEMA        |  (self.MEMA, r'%s的%s周期平滑指数移动平均', 2),                  |   |   |   |
| EXPMEMA     |  (self.MEMA, r'%s的%s周期平滑指数移动平均', 2),                  |   |   |   |
| DMA         |   (self.DMA, r'%s的%s周期动态平均', 2),                          |   |   |   |
| SMA         |   (self.SMA, r'%s的%s周期(权重                                   |   |   |   |
| CONV        |   (self.CONV, r'%s与%s的%s周期卷积', 3),                         |   |   |   |
| SAR         |   (self.SAR, r'周期为%s步长为%s极值为%s的抛物转向', 3),          |   |   |   |
| SLOPE       |  (self.SLOPE, r'%s的周期为%s的线性回归斜率', 2),                 |   |   |   |
| CLAMP       |  (self.CLAMP, r'限定%s的输出在(%s, %s)之间', 3),                 |   |   |   |
| FORCAST     |  (self.FORCAST, r'%s的周期为%s的线性预测', 2),                   |   |   |   |
| DRAWFUNC    |  (self.DRAWFUNC, r'DRAWFUNC(%s, %s, %s)', 3),                    |   |   |   |
| DRAWICON    |  (self.DRAWICON, r'DRAWICON(%s, %s, %s)', 3),                    |   |   |   |
| DRAWICONF   |  (self.DRAWICONF, r'DRAWICONF(%s, %s, %s)', 3),                  |   |   |   |
| STICKLINE   |  (self.STICKLINE, r'STICKLINE(%s,%s,%s,%s,%s)', 5),              |   |   |   |
| DRAWKLINE   |  (self.DRAWKLINE, r'DRAWKLINE(%s, %s, %s, %s)', 4),              |   |   |   |
| DRAWSKLINE  |  (self.DRAWSKLINE, r'DRAWSKLINE(%s, %s, %s, %s)', 4),            |   |   |   |
| DRAWPKLINE  |  (self.DRAWPKLINE, r'DRAWPKLINE(%s, %s, %s, %s, %s)', 5),        |   |   |   |
| DRAWNUMBER  |  (self.DRAWNUMBER, r'DRAWNUMBER(%s, %s, %s)', 3),                |   |   |   |
| DRAWTEXT    |  (self.DRAWTEXT, r'DRAWTEXT(%s, %s, %s)', 3),                    |   |   |   |
| DRAWNULL    |  (np.NaN, r'DRAWNULL', 0),                                       |   |   |   |
| DRAWGRID    |  (self.DRAWGRID, r'DRAWGRID(%s)', 1),                            |   |   |   |
| DRAWVOL     |  (self.DRAWVOL, r'DRAWVOL(%s, %s)', 2),                          |   |   |   |
| SIGNAL      |  (self.SIGNAL, r'从仓位数据%s导出指定的买入%s卖出%s信号指示',3), |   |   |   |
| BASE        |  (self.BASE, r'创建%s的基准', 1),                                |   |   |   |
| ASSET       |  (self.ASSET, r'根据价格%s,仓位%s和建仓比例%s创建资产', 3),      |   |   |   |
| SUGAR       |  (self.SUGAR, r'根据盈利情况进行调仓操作(仓位                    |   |   |   |
| REMEMB      |  (self.REMEMB, r'记录价格(条件                                   |   |   |   |
| STOP        | (self.STOP, r'止盈止损点(条件                                    |   |   |   |
| HORIZON     | (self.HORIZON, r'%s的周期为%s的地平线', 2)                       |   |   |   |

# Bug reports

Hope this small piece of code does your help, if it can make your code more simple, more maintainability, I shall feel happy. If you find any problems, or you have any improvement advice, please contact with me by the following e-mail address:

-- By woody(li.woodyli@gmail.com)




