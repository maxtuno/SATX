"""
///////////////////////////////////////////////////////////////////////////////
//        Copyright (c) 2012-2021 Oscar Riveros. all rights reserved.        //
//                        oscar.riveros@peqnp.science                        //
//                                                                           //
//   without any restriction, Oscar Riveros reserved rights, patents and     //
//  commercialization of this knowledge or derived directly from this work.  //
///////////////////////////////////////////////////////////////////////////////

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import sys
import setuptools
from distutils.core import setup, Extension

ext_modules = [
    Extension("slime",
              language="c++",
              sources=['src/SLIME.cc', 'src/SimpSolver.cc', 'src/Solver.cc'],
              include_dirs=['.', 'include'],
              extra_compile_args=['-std=c++11'],
              ),
    Extension("pixie",
              language="c++",
              sources=['src/pixie.cc'],
              include_dirs=['.', 'include'],
              extra_compile_args=['-std=c++11'],
              ),
]
setup(
    name='SATX',
    version='0.1.1',
    packages=['satx'],
    url='http://www.peqnp.com',
    license='copyright (c) 2012-2021 Oscar Riveros. All rights reserved.',
    author='Oscar Riveros',
    author_email='contact@peqnp.com',
    description='PEQNP Mathematical Solver from http://www.peqnp.com',
    ext_modules=ext_modules,
)
