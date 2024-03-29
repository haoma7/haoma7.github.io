


var swiper = new Swiper( '.swiper-container.two', {
  pagination: '.swiper-pagination',
  paginationClickable: true,
    effect: 'coverflow',
    centeredSlides: true,
    slidesPerView: 'auto',
    initialSlide: 1,


    coverflow: {
        rotate: 0,
        stretch: 50,
        depth: 150,
        scale:5,
        modifier: 1.5,
        slideShadows:true,
    },
    navigation:{
      nextEl: ".swiper-button-next",
      prevEl: ".swiper-button-prev"
    },
    keyboard: {
      enabled: true,
    },
} );


swiper.on("keyPress", (swiper, keyCode) => {
  switch (keyCode) {
    case 38:
      swiper.slidePrev();
      break;
    case 40:
      swiper.slideNext();
      break;
  }
});
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