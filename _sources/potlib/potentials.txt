Ready made Potentials
---------------------

The following sections contain some potentials that are implemented in the potential
library. The plots show the eigenvalues or energy surfaces. Some potentials
have additional parameters, the default values for these are also Name.

Potential ``cos_osc``
^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = a \left(- \cos{\left (b x \right )} + 1\right)`

* Variables: :math:`x`

* Default values:

  * :math:`a = 0.07`
  * :math:`b = 1.0`

.. image:: fig/cos_osc.png
   :width: 400px

Potential ``cosh_osc``
^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = a \cosh{\left (b x \right )}`

* Variables: :math:`x`

* Default values:

  * :math:`a = 1`
  * :math:`b = 1`

.. image:: fig/cosh_osc.png
   :width: 400px

Potential ``double_well``
^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \sigma \left(x^{2} - 1\right)^{2}`

* Variables: :math:`x`

* Default values:

  * :math:`\sigma = 1.0`

.. image:: fig/double_well.png
   :width: 400px

Potential ``double_well2``
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = a x^{4} - b x^{2}`

* Variables: :math:`x`

* Default values:

  * :math:`a = 1.0`
  * :math:`b = 1.0`

.. image:: fig/double_well2.png
   :width: 400px

Potential ``eckart``
^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \frac{\sigma}{\cosh^{2}{\left (\frac{x}{a} \right )}}`

* Variables: :math:`x`

* Default values:

  * :math:`a = 0.944858082316`
  * :math:`\sigma = 0.038088`

.. image:: fig/eckart.png
   :width: 400px

Potential ``free_particle``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = c`

* Variables: :math:`x`

* Default values:

  * :math:`c = 0`

.. image:: fig/free_particle.png
   :width: 400px

Potential ``kratzer``
^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \frac{b \left(b - 1\right)}{2 x^{2}} + \frac{x^{2}}{2}`

* Variables: :math:`x`

* Default values:

  * :math:`b = 2.0`

.. image:: fig/kratzer.png
   :width: 400px

Potential ``morse``
^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = D \left(1 - e^{- a \left(x - x_{0}\right)}\right)^{2}`

* Variables: :math:`x`

* Default values:

  * :math:`a = 0.5`
  * :math:`x_{0} = 0.0`
  * :math:`D = 3.0`

.. image:: fig/morse.png
   :width: 400px

Potential ``morse_zero``
^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = D \left(- 2 e^{- a \left(x - x_{0}\right)} + e^{- 2 a \left(x - x_{0}\right)}\right)`

* Variables: :math:`x`

* Default values:

  * :math:`a = 0.5`
  * :math:`x_{0} = 0.0`
  * :math:`D = 3.0`

.. image:: fig/morse_zero.png
   :width: 400px

Potential ``morse_zero_2``
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = l^{2} \left(e^{- 2 x + 2 x_{0}} - 2 e^{- x + x_{0}}\right)`

* Variables: :math:`x`

* Default values:

  * :math:`x_{0} = 0.0`
  * :math:`l = 1.0`

.. image:: fig/morse_zero_2.png
   :width: 400px

Potential ``pert_quadratic``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \frac{\delta^{2} x^{2}}{2} + \frac{\sigma x^{2}}{2}`

* Variables: :math:`x`

* Default values:

  * :math:`\sigma = 0.05`
  * :math:`\delta = 0.2`

.. image:: fig/pert_quadratic.png
   :width: 400px

Potential ``quadratic``
^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \frac{\sigma x^{2}}{2}`

* Variables: :math:`x`

* Default values:

  * :math:`\sigma = 1/2`

.. image:: fig/quadratic.png
   :width: 400px

Potential ``quartic``
^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \frac{\sigma x^{4}}{4}`

* Variables: :math:`x`

* Default values:

  * :math:`\sigma = 0.05`

.. image:: fig/quartic.png
   :width: 400px

Potential ``v_shape``
^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \frac{1}{2} \sqrt{4 \delta^{2} + \tanh^{2}{\left (x \right )}}`

* Variables: :math:`x`

* Default values:

  * :math:`\delta = 0.2`

.. image:: fig/v_shape.png
   :width: 400px

Potential ``wall``
^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \operatorname{atan}{\left (\sigma x \right )} + \frac{\pi}{2}`

* Variables: :math:`x`

* Default values:

  * :math:`\sigma = 10.0`

.. image:: fig/wall.png
   :width: 400px

Potential ``delta_gap``
^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \left[\begin{matrix}\frac{1}{2} \tanh{\left (x \right )} & \delta\\\delta & - \frac{1}{2} \tanh{\left (x \right )}\end{matrix}\right]`

* Variables: :math:`x`

* Default values:

  * :math:`\delta = 0.2`

.. image:: fig/delta_gap.png
   :width: 400px

Potential ``delta_gap_diag``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \left[\begin{matrix}\sqrt{\delta^{2} + \frac{1}{4} \tanh^{2}{\left (x \right )}} & 0\\0 & - \sqrt{\delta^{2} + \frac{1}{4} \tanh^{2}{\left (x \right )}}\end{matrix}\right]`

* Variables: :math:`x`


.. image:: fig/delta_gap_diag.png
   :width: 400px

Potential ``two_crossings``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \left[\begin{matrix}\frac{1}{2} \tanh{\left (- \rho + x \right )} \tanh{\left (\rho + x \right )} & \frac{\delta}{2}\\\frac{\delta}{2} & - \frac{1}{2} \tanh{\left (- \rho + x \right )} \tanh{\left (\rho + x \right )}\end{matrix}\right]`

* Variables: :math:`x`

* Default values:

  * :math:`\rho = 3.0`

.. image:: fig/two_crossings.png
   :width: 400px

Potential ``two_quadratic``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \left[\begin{matrix}\frac{\sigma x^{2}}{2} & 0\\0 & \frac{\sigma x^{2}}{2}\end{matrix}\right]`

* Variables: :math:`x`

* Default values:

  * :math:`\sigma = 0.05`

.. image:: fig/two_quadratic.png
   :width: 400px

Potential ``two_quartic``
^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \left[\begin{matrix}\frac{\sigma x^{4}}{4} & 0\\0 & \frac{\sigma x^{4}}{8}\end{matrix}\right]`

* Variables: :math:`x`

* Default values:

  * :math:`\sigma = 1`

.. image:: fig/two_quartic.png
   :width: 400px

Potential ``three_levels``
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \left[\begin{matrix}\tanh{\left (- \rho + x \right )} + \tanh{\left (\rho + x \right )} & \delta_{1} & \delta_{2}\\\delta_{1} & - \tanh{\left (\rho + x \right )} & 0\\\delta_{2} & 0 & - \tanh{\left (- \rho + x \right )} + 1\end{matrix}\right]`

* Variables: :math:`x`

* Default values:

  * :math:`\rho = 3.0`

.. image:: fig/three_levels.png
   :width: 400px

Potential ``three_quadratic``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \left[\begin{matrix}\frac{\sigma x^{2}}{2} & 0 & 0\\0 & \frac{\sigma x^{2}}{2} & 0\\0 & 0 & \frac{\sigma x^{2}}{2}\end{matrix}\right]`

* Variables: :math:`x`

* Default values:

  * :math:`\sigma = 0.05`

.. image:: fig/three_quadratic.png
   :width: 400px

Potential ``four_powers``
^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \left[\begin{matrix}\frac{\sigma x^{2}}{2} & 0 & 0 & 0\\0 & \frac{\sigma x^{4}}{4} & 0 & 0\\0 & 0 & \frac{\sigma x^{6}}{6} & 0\\0 & 0 & 0 & \frac{\sigma x^{8}}{8}\end{matrix}\right]`

* Variables: :math:`x`

* Default values:

  * :math:`\sigma = 0.05`

.. image:: fig/four_powers.png
   :width: 400px

Potential ``four_quadratic``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \left[\begin{matrix}\frac{\sigma x^{2}}{2} & 0 & 0 & 0\\0 & \frac{\sigma x^{2}}{2} & 0 & 0\\0 & 0 & \frac{\sigma x^{2}}{2} & 0\\0 & 0 & 0 & \frac{\sigma x^{2}}{2}\end{matrix}\right]`

* Variables: :math:`x`

* Default values:

  * :math:`\sigma = 0.05`

.. image:: fig/four_quadratic.png
   :width: 400px

Potential ``five_quadratic``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \left[\begin{matrix}\frac{\sigma x^{2}}{2} & 0 & 0 & 0 & 0\\0 & \frac{\sigma x^{2}}{2} & 0 & 0 & 0\\0 & 0 & \frac{\sigma x^{2}}{2} & 0 & 0\\0 & 0 & 0 & \frac{\sigma x^{2}}{2} & 0\\0 & 0 & 0 & 0 & \frac{\sigma x^{2}}{2}\end{matrix}\right]`

* Variables: :math:`x`

* Default values:

  * :math:`\sigma = 0.05`

.. image:: fig/five_quadratic.png
   :width: 400px

Potential ``channel_2d``
^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = sigmax x + \frac{sigmay y^{2}}{2}`

* Variables: :math:`x`, :math:`y`

* Default values:

  * :math:`sigmay = 0.45`
  * :math:`sigmax = 0.0`

.. image:: fig/channel_2d.png
   :width: 400px

Potential ``circle_pit_2d``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \operatorname{atan}{\left (\sigma \left(- R + \sqrt{x^{2} + y^{2}}\right) \right )} + \frac{\pi}{2}`

* Variables: :math:`x`, :math:`y`

* Default values:

  * :math:`R = 8`
  * :math:`\sigma = 10`

.. image:: fig/circle_pit_2d.png
   :width: 400px

Potential ``corral_ring``
^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = - \frac{1}{2} \sqrt{\delta^{2} + \tanh^{2}{\left (- R + \sqrt{x^{2} + y^{2}} \right )} \tanh^{2}{\left (R + \sqrt{x^{2} + y^{2}} \right )}}`

* Variables: :math:`x`, :math:`y`

* Default values:

  * :math:`R = 3`
  * :math:`\delta = 1`

.. image:: fig/corral_ring.png
   :width: 400px

Potential ``corral_rotsym_2d``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \operatorname{atan}{\left (\sigma \left(- R + \sqrt{x^{2} + y^{2}}\right) \right )} + \frac{\pi}{2}`

* Variables: :math:`x`, :math:`y`

* Default values:

  * :math:`R = 8`
  * :math:`\sigma = 10`

.. image:: fig/corral_rotsym_2d.png
   :width: 400px

Potential ``cos_osc_2d``
^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = ax \left(- \cos{\left (bx x \right )} + 1\right) + ay \left(- \cos{\left (by y \right )} + 1\right)`

* Variables: :math:`x`, :math:`y`

* Default values:

  * :math:`ay = 1`
  * :math:`ax = 1`
  * :math:`bx = 1`
  * :math:`by = 1`

.. image:: fig/cos_osc_2d.png
   :width: 400px

Potential ``cos_osc_add_2d``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = - \cos{\left (a x \right )} - \cos{\left (b y \right )}`

* Variables: :math:`x`, :math:`y`

* Default values:

  * :math:`a = 1`
  * :math:`b = 1`

.. image:: fig/cos_osc_add_2d.png
   :width: 400px

Potential ``cos_osc_mul_2d``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = - \cos{\left (a x \right )} \cos{\left (b y \right )}`

* Variables: :math:`x`, :math:`y`

* Default values:

  * :math:`a = 1`
  * :math:`b = 1`

.. image:: fig/cos_osc_mul_2d.png
   :width: 400px

Potential ``cos_osc_rotsym_2d``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = - a \cos{\left (b \left(x^{2} + y^{2}\right) \right )}`

* Variables: :math:`x`, :math:`y`

* Default values:

  * :math:`a = 1`
  * :math:`b = 1`

.. image:: fig/cos_osc_rotsym_2d.png
   :width: 400px

Potential ``cosh_osc_2d``
^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = ax \left(\cosh{\left (bx x \right )} + 1\right) + ay \left(\cosh{\left (by y \right )} + 1\right)`

* Variables: :math:`x`, :math:`y`

* Default values:

  * :math:`ay = 1`
  * :math:`ax = 1`
  * :math:`bx = 1`
  * :math:`by = 1`

.. image:: fig/cosh_osc_2d.png
   :width: 400px

Potential ``cosh_osc_add_2d``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \cosh{\left (a x \right )} + \cosh{\left (b y \right )}`

* Variables: :math:`x`, :math:`y`

* Default values:

  * :math:`a = 1`
  * :math:`b = 1`

.. image:: fig/cosh_osc_add_2d.png
   :width: 400px

Potential ``cosh_osc_mul_2d``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \cosh{\left (a x \right )} \cosh{\left (b y \right )}`

* Variables: :math:`x`, :math:`y`

* Default values:

  * :math:`a = 1`
  * :math:`b = 1`

.. image:: fig/cosh_osc_mul_2d.png
   :width: 400px

Potential ``cosh_osc_rotsym_2d``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = a \cosh{\left (b \sqrt{x^{2} + y^{2}} \right )}`

* Variables: :math:`x`, :math:`y`

* Default values:

  * :math:`a = 1`
  * :math:`b = 1`

.. image:: fig/cosh_osc_rotsym_2d.png
   :width: 400px

Potential ``double_well_2d``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = ax x^{4} + ay y^{4} - bx x^{2} - by y^{2} - cx x - cy y`

* Variables: :math:`x`, :math:`y`

* Default values:

  * :math:`cy = 0.0`
  * :math:`cx = 0.0`
  * :math:`ay = 1.0`
  * :math:`ax = 1.0`
  * :math:`bx = 4.0`
  * :math:`by = 0.0`

.. image:: fig/double_well_2d.png
   :width: 400px

Potential ``double_well_harmonic_2d``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = ax x^{4} + ay y^{4} - bx x^{2} - by y^{2} - cx x - cy y`

* Variables: :math:`x`, :math:`y`

* Default values:

  * :math:`cy = 0.0`
  * :math:`cx = 0.0`
  * :math:`ay = 0.0`
  * :math:`ax = 1.0`
  * :math:`bx = 4.0`
  * :math:`by = -1.0`

.. image:: fig/double_well_harmonic_2d.png
   :width: 400px

Potential ``eckart_bn``
^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \frac{k y^{2}}{2} \left(- \sigma e^{- l x^{2}} + 1\right) + \frac{v_{0}}{\cosh^{2}{\left (a x \right )}}`

* Variables: :math:`x`, :math:`y`

* Default values:

  * :math:`v_{0} = 0.425`
  * :math:`a = 1.3624`
  * :math:`k = 0.06784`
  * :math:`\sigma = 0.5`
  * :math:`l = 0.25`

.. image:: fig/eckart_bn.png
   :width: 400px

Potential ``gauss_hill_2d``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = e^{- sigmax x^{2} - sigmay y^{2}}`

* Variables: :math:`x`, :math:`y`

* Default values:

  * :math:`sigmay = 1`
  * :math:`sigmax = 1`

.. image:: fig/gauss_hill_2d.png
   :width: 400px

Potential ``harmonic_channel``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \sigma y + \frac{w^{2} x^{2}}{2}`

* Variables: :math:`x`, :math:`y`

* Default values:

  * :math:`\sigma = -0.1`
  * :math:`w = 1.0`

.. image:: fig/harmonic_channel.png
   :width: 400px

Potential ``henon_heiles``
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \frac{a}{2} \left(x^{2} + y^{2}\right) + b \left(x^{2} y - \frac{y^{3}}{3}\right)`

* Variables: :math:`x`, :math:`y`

* Default values:

  * :math:`a = 1`
  * :math:`b = 1/2`

.. image:: fig/henon_heiles.png
   :width: 400px

Potential ``morse_threefold``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \left(- e^{\left(- \sigma - \frac{1}{16} \left(- \cos{\left (3 \operatorname{atan_{2}}{\left (y,x \right )} \right )} + 1\right)^{2}\right) \left(x^{2} + y^{2}\right)} + 1\right)^{2}`

* Variables: :math:`x`, :math:`y`

* Default values:

  * :math:`\sigma = 0.05`

.. image:: fig/morse_threefold.png
   :width: 400px

Potential ``morse_threefold_2``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \left(- e^{\frac{1}{16 \left(x^{2} + y^{2}\right)^{2}} \left(- 16 \sigma \left(x^{2} + y^{2}\right)^{3} - \left(x \left(x^{2} - 3 y^{2}\right) - \left(x^{2} + y^{2}\right)^{\frac{3}{2}}\right)^{2}\right)} + 1\right)^{2}`

* Variables: :math:`x`, :math:`y`

* Default values:

  * :math:`\sigma = 0.05`

.. image:: fig/morse_threefold_2.png
   :width: 400px

Potential ``pullen_edmonds``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = a x^{2} y^{2} + \frac{b_{1} x^{2}}{2} + \frac{b_{2} y^{2}}{2}`

* Variables: :math:`x`, :math:`y`

* Default values:

  * :math:`a = 1`
  * :math:`b_{1} = 1`
  * :math:`b_{2} = 1`

.. image:: fig/pullen_edmonds.png
   :width: 400px

Potential ``quad_well``
^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = ax x^{4} + ay y^{4} - bx x^{2} - by y^{2} - cx x - cy y`

* Variables: :math:`x`, :math:`y`

* Default values:

  * :math:`cy = 0.0`
  * :math:`cx = 0.0`
  * :math:`ay = 1.0`
  * :math:`ax = 1.0`
  * :math:`bx = 3.0`
  * :math:`by = 3.0`

.. image:: fig/quad_well.png
   :width: 400px

Potential ``quadratic_2d``
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \frac{sigmax x^{2}}{2} + \frac{sigmay y^{2}}{2}`

* Variables: :math:`x`, :math:`y`

* Default values:

  * :math:`sigmay = 1/2`
  * :math:`sigmax = 1/2`

.. image:: fig/quadratic_2d.png
   :width: 400px

Potential ``quartic_2d``
^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = sigmax x^{4} + sigmay y^{4}`

* Variables: :math:`x`, :math:`y`

* Default values:

  * :math:`sigmay = 1`
  * :math:`sigmax = 1`

.. image:: fig/quartic_2d.png
   :width: 400px

Potential ``quartic_2d_rotsym``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = sigmax^{2} x^{4} + 2 sigmax sigmay x^{2} y^{2} + sigmay^{2} y^{4}`

* Variables: :math:`x`, :math:`y`

* Default values:

  * :math:`sigmay = 1`
  * :math:`sigmax = 1`

.. image:: fig/quartic_2d_rotsym.png
   :width: 400px

Potential ``quartic_reg_2d``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = sigmax x^{4} + sigmay y^{4} + taux x^{2} + tauy y^{2}`

* Variables: :math:`x`, :math:`y`

* Default values:

  * :math:`taux = 1`
  * :math:`tauy = 1`
  * :math:`sigmay = 1`
  * :math:`sigmax = 1`

.. image:: fig/quartic_reg_2d.png
   :width: 400px

Potential ``quartic_rotsym_reg_2d``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = sigmax x^{4} + sigmay y^{4} + taux x^{2} + tauy y^{2} + 2 x^{2} y^{2} \sqrt{sigmax sigmay}`

* Variables: :math:`x`, :math:`y`

* Default values:

  * :math:`taux = 1`
  * :math:`tauy = 1`
  * :math:`sigmay = 1`
  * :math:`sigmax = 1`

.. image:: fig/quartic_rotsym_reg_2d.png
   :width: 400px

Potential ``ring_valley``
^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \frac{1}{2} \sqrt{\delta^{2} + \tanh^{2}{\left (- R + \sqrt{x^{2} + y^{2}} \right )} \tanh^{2}{\left (R + \sqrt{x^{2} + y^{2}} \right )}}`

* Variables: :math:`x`, :math:`y`

* Default values:

  * :math:`R = 3`
  * :math:`\delta = 1`

.. image:: fig/ring_valley.png
   :width: 400px

Potential ``sextic_2d``
^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = sigmax x^{6} + sigmay y^{6}`

* Variables: :math:`x`, :math:`y`

* Default values:

  * :math:`sigmay = 1`
  * :math:`sigmax = 1`

.. image:: fig/sextic_2d.png
   :width: 400px

Potential ``sextic_reg_2d``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = sigmax x^{6} + sigmay y^{6} + taux x^{2} + tauy y^{2}`

* Variables: :math:`x`, :math:`y`

* Default values:

  * :math:`taux = 1`
  * :math:`tauy = 1`
  * :math:`sigmay = 1`
  * :math:`sigmax = 1`

.. image:: fig/sextic_reg_2d.png
   :width: 400px

Potential ``sextic_rotsym_reg_2d``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = ax^{3} x^{6} + 3 ax^{2} ay x^{4} y^{2} + 3 ax ay^{2} x^{2} y^{4} + ay^{3} y^{6} + taux x^{2} + tauy y^{2}`

* Variables: :math:`x`, :math:`y`

* Default values:

  * :math:`ay = 1`
  * :math:`ax = 1`
  * :math:`taux = 1`
  * :math:`tauy = 1`

.. image:: fig/sextic_rotsym_reg_2d.png
   :width: 400px

Potential ``sine_maar``
^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \alpha e^{- \sigma \left(x^{2} + y^{2}\right)} + \sin{\left (x^{2} + y^{2} \right )}`

* Variables: :math:`x`, :math:`y`

* Default values:

  * :math:`\alpha = 0.8`
  * :math:`\sigma = 1.0`

.. image:: fig/sine_maar.png
   :width: 400px

Potential ``conic``
^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \left[\begin{matrix}x & y\\y & - x\end{matrix}\right]`

* Variables: :math:`x`, :math:`y`


.. image:: fig/conic.png
   :width: 400px

Potential ``conic_avoided``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \left[\begin{matrix}x & \sqrt{\delta^{2} + y^{2}}\\\sqrt{\delta^{2} + y^{2}} & - x\end{matrix}\right]`

* Variables: :math:`x`, :math:`y`

* Default values:

  * :math:`\delta = 1.0`

.. image:: fig/conic_avoided.png
   :width: 400px

Potential ``conic_avoided_c``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \left[\begin{matrix}x & i \delta + y\\- i \delta + y & - x\end{matrix}\right]`

* Variables: :math:`x`, :math:`y`


.. image:: fig/conic_avoided_c.png
   :width: 400px

Potential ``delta_gap_rotsym``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \left[\begin{matrix}\frac{1}{2} \tanh{\left (\sqrt{x^{2} + y^{2}} \right )} & \delta\\\delta & - \frac{1}{2} \tanh{\left (\sqrt{x^{2} + y^{2}} \right )}\end{matrix}\right]`

* Variables: :math:`x`, :math:`y`


.. image:: fig/delta_gap_rotsym.png
   :width: 400px

Potential ``harmonic_tube``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \sigma z + \frac{wx^{2} x^{2}}{2} + \frac{wy^{2} y^{2}}{2}`

* Variables: :math:`x`, :math:`y`, :math:`z`

* Default values:

  * :math:`\sigma = -0.1`
  * :math:`wy = 1.0`
  * :math:`wx = 1.0`

Potential ``quadratic_3d``
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \frac{sigmax x^{2}}{2} + \frac{sigmay y^{2}}{2} + \frac{sigmaz z^{2}}{2}`

* Variables: :math:`x`, :math:`y`, :math:`z`

* Default values:

  * :math:`sigmay = 1/2`
  * :math:`sigmax = 1/2`
  * :math:`sigmaz = 1/2`

Potential ``harmonic_hypertube``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \sigma x_{4} + \frac{w_{1}^{2} x_{1}^{2}}{2} + \frac{w_{2}^{2} x_{2}^{2}}{2} + \frac{w_{3}^{2} x_{3}^{2}}{2}`

* Variables: :math:`x_{1}`, :math:`x_{2}`, :math:`x_{3}`, :math:`x_{4}`

* Default values:

  * :math:`\sigma = -0.1`
  * :math:`w_{3} = 1.0`
  * :math:`w_{2} = 1.0`
  * :math:`w_{1} = 1.0`

Potential ``quadratic_4d``
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Formula: :math:`V(x) = \frac{\sigma_{1}}{2} + \frac{\sigma_{2} x_{2}^{2}}{2} + \frac{\sigma_{3} x_{3}^{2}}{2} + \frac{\sigma_{4} x_{4}^{2}}{2}`

* Variables: :math:`x_{1}`, :math:`x_{2}`, :math:`x_{3}`, :math:`x_{4}`

* Default values:

  * :math:`\sigma_{4} = 1/2`
  * :math:`\sigma_{1} = 1/2`
  * :math:`\sigma_{3} = 1/2`
  * :math:`\sigma_{2} = 1/2`

