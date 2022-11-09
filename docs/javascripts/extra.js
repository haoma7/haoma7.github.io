var swiper = new Swiper( '.swiper-container.two', {
   
    effect: 'coverflow',
    centeredSlides: true,
    slidesPerView: 'auto',
    coverflow: {
        rotate: 0,
        stretch: 50,
        depth: 150,
        scale:5,
        modifier: 1.5,
        slideShadows:true,
    }
} );

const myAtropos = Atropos({
    el: '.my-atropos',
    activeOffset: 40,
    shadowScale: 1.05,
    onEnter() {
      console.log('Enter');
    },
    onLeave() {
      console.log('Leave');
    },
    onRotate(x, y) {
      console.log('Rotate', x, y);
    }
  });